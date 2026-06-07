# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa
import yaml
from datus_storage_base.conditions import And, WhereExpr, eq, in_

from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)


class SemanticModelStorage(BaseEmbeddingStore):
    """Storage for field-level semantic objects (tables, columns) - excluding metrics."""

    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        super().__init__(
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
                    # -- Column Expression & Type --
                    pa.field("expr", pa.string()),  # SQL expression (e.g., "amount * quantity")
                    pa.field(
                        "column_type", pa.string()
                    ),  # Dim: CATEGORICAL|TIME; Ident: PRIMARY|FOREIGN|UNIQUE|NATURAL
                    # -- Measure Specific --
                    pa.field("agg", pa.string()),  # SUM|COUNT|COUNT_DISTINCT|AVERAGE|MIN|MAX|PERCENTILE|MEDIAN
                    pa.field("create_metric", pa.bool_()),  # Auto-create metric flag
                    pa.field("agg_time_dimension", pa.string()),  # Aggregation time dimension
                    # -- Dimension Specific --
                    pa.field("is_partition", pa.bool_()),  # Partition column flag
                    pa.field("time_granularity", pa.string()),  # For TIME dims: DAY|WEEK|MONTH|QUARTER|YEAR
                    # -- Identifier Specific --
                    pa.field("entity", pa.string()),  # Associated entity name
                    # -- Operations & Lineage --
                    pa.field("yaml_path", pa.string()),
                    pa.field("updated_at", pa.timestamp("ms")),
                ]
            ),
            vector_source_name="description",
            vector_column_name="vector",
            unique_columns=["id"],
            **kwargs,
        )

    def create_indices(self):
        self._ensure_table_ready()

        self._create_scalar_index("kind")
        self._create_scalar_index("table_name")
        self._create_scalar_index("id")

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

        where_clause = And(conditions) if conditions else None

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

    # ------------------------------------------------------------------
    # Field mappings: vector DB field → YAML key
    # ------------------------------------------------------------------
    _TABLE_DB_TO_YAML: Dict[str, str] = {"description": "description"}
    _COLUMN_DB_TO_YAML: Dict[str, str] = {
        "description": "description",
        "expr": "expr",
        "column_type": "type",
        "agg": "agg",
        "create_metric": "create_metric",
        "agg_time_dimension": "agg_time_dimension",
        "is_partition": "is_partition",
        "time_granularity": "time_granularity",
        "entity": "entity",
    }

    def update_entry(self, entry_id: str, update_values: Dict[str, Any]) -> bool:
        """Update a semantic model entry by ID and sync changes to the YAML file.

        Args:
            entry_id: Unique entry ID (e.g., "table:orders", "column:orders.amount")
            update_values: Dictionary of field names and new values

        Returns:
            True if update successful

        Raises:
            ValueError: If entry not found or update_values is empty
        """
        if not entry_id:
            raise DatusException(
                ErrorCode.STORAGE_INVALID_ARGUMENT, message_args={"error_message": "entry_id must not be empty"}
            )
        if not update_values:
            raise DatusException(
                ErrorCode.STORAGE_INVALID_ARGUMENT, message_args={"error_message": "update_values must not be empty"}
            )

        entries = self._search_all(
            where=eq("id", entry_id), select_fields=["id", "kind", "name", "table_name", "yaml_path"]
        ).to_pylist()
        if not entries:
            raise DatusException(ErrorCode.STORAGE_ENTRY_NOT_FOUND, message_args={"entry_id": entry_id})
        entry = entries[0]
        yaml_path = entry.get("yaml_path", "")
        # Use table_name to disambiguate the data_source document when a YAML file holds
        # multiple semantic models. For table-kind rows the entry's own ``name`` IS the
        # data_source name, so fall back to it when ``table_name`` is empty.
        data_source_name = entry.get("table_name") or entry["name"]

        self.update(where=eq("id", entry_id), update_values=update_values)

        if yaml_path:
            self._sync_semantic_update_to_yaml(yaml_path, entry["kind"], entry["name"], data_source_name, update_values)

        logger.info(f"Updated semantic model entry '{entry['name']}' (kind={entry['kind']})")
        return True

    def _sync_semantic_update_to_yaml(
        self,
        yaml_path: str,
        kind: str,
        name: str,
        data_source_name: str,
        update_values: Dict[str, Any],
    ) -> None:
        """Sync update_values for a semantic model entry back to its YAML file.

        Args:
            yaml_path: Path to the YAML file containing the data_source document
            kind: Entry kind — "table" or "column"
            name: Short name of the entry (table name or column name)
            data_source_name: Name of the parent data_source document (used to disambiguate
                YAML files that contain multiple ``data_source`` blocks)
            update_values: Dictionary of vector-DB field names and new values
        """
        if not os.path.exists(yaml_path):
            return

        try:
            with open(yaml_path, encoding="utf-8") as f:
                docs = [doc for doc in yaml.safe_load_all(f) if doc is not None]

            # Prefer an exact match on the data_source name; only fall back to the first
            # data_source doc if no name match exists (preserves legacy single-doc behavior
            # without silently mutating the wrong model in a multi-doc file).
            data_source = None
            fallback_data_source = None
            for doc in docs:
                ds = doc.get("data_source") if isinstance(doc, dict) else None
                if not isinstance(ds, dict):
                    continue
                if fallback_data_source is None:
                    fallback_data_source = ds
                if data_source_name and ds.get("name") == data_source_name:
                    data_source = ds
                    break
            if data_source is None:
                # Only fall back when the file holds exactly one data_source doc; otherwise
                # we cannot safely guess which model to mutate.
                ds_count = sum(1 for d in docs if isinstance(d, dict) and isinstance(d.get("data_source"), dict))
                if ds_count == 1:
                    data_source = fallback_data_source

            if data_source is None:
                return

            updated = False
            if kind == "table":
                for db_key, yaml_key in self._TABLE_DB_TO_YAML.items():
                    if db_key in update_values:
                        data_source[yaml_key] = update_values[db_key]
                        updated = True

            elif kind == "column":
                target_item = None
                for section in ("dimensions", "measures", "identifiers"):
                    for item in data_source.get(section, []):
                        if item.get("name") == name:
                            target_item = item
                            break
                    if target_item is not None:
                        break

                if target_item is not None:
                    for db_key, yaml_key in self._COLUMN_DB_TO_YAML.items():
                        if db_key in update_values:
                            target_item[yaml_key] = update_values[db_key]
                            updated = True

            if not updated:
                return

            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)

            logger.info(f"Updated semantic model entry '{name}' (kind={kind}) in yaml file: {yaml_path}")

        except Exception as e:
            logger.error(f"Failed to update yaml file {yaml_path}: {e}")


class SemanticModelRAG:
    """RAG interface for semantic model operations.

    Handles datasource_id filtering on reads and field injection on writes.
    """

    def __init__(
        self,
        agent_config: "AgentConfig",
        sub_agent_name: Optional[str] = None,
        datasource_id: Optional[str] = None,
    ):
        from datus.storage.rag_scope import _build_sub_agent_filter
        from datus.storage.registry import get_storage

        self.datasource_id = datasource_id or agent_config.current_datasource or ""
        self.storage: SemanticModelStorage = get_storage(
            SemanticModelStorage, "semantic_model", project=agent_config.project_name
        )
        self._sub_agent_filter = _build_sub_agent_filter(agent_config, sub_agent_name, self.storage, "tables")

    def _sub_agent_conditions(self) -> list:
        """Build sub-agent filter conditions (datasource_id handled by backend)."""
        conditions = []
        if self._sub_agent_filter:
            conditions.append(self._sub_agent_filter)
        return conditions

    def truncate(self) -> None:
        """Delete all semantic model data for this datasource."""
        self.storage.truncate_scoped()

    def get_semantic_model(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        select_fields: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct semantic model object from granular storage."""
        if not table_name:
            logger.warning("get_semantic_model called without table_name")
            return None

        base_conds = self._sub_agent_conditions()

        # Build filter conditions
        table_conds = [eq("kind", "table"), eq("table_name", table_name)] + base_conds
        if catalog_name:
            table_conds.append(eq("catalog_name", catalog_name))
        if database_name:
            table_conds.append(eq("database_name", database_name))
        if schema_name:
            table_conds.append(eq("schema_name", schema_name))

        table_objs = self.storage._search_all(where=And(table_conds)).to_pylist()

        # Fallback 1: broad match
        if not table_objs and (catalog_name or database_name or schema_name):
            logger.debug(f"Semantic model not found for {table_name} with full filters, trying broad match.")
            broad_conds = [
                eq("kind", "table"),
                eq("table_name", table_name),
            ] + base_conds
            table_objs = self.storage._search_all(where=And(broad_conds)).to_pylist()

        # Fallback 2: case-insensitive
        if not table_objs:
            if table_name.lower() != table_name:
                lower_conds = [
                    eq("kind", "table"),
                    eq("table_name", table_name.lower()),
                ] + base_conds
                table_objs = self.storage._search_all(where=And(lower_conds)).to_pylist()

        if not table_objs:
            return None

        semantic_model = table_objs[0]
        model_name = semantic_model.get("name", table_name)

        # Find children
        children_conds = [
            eq("kind", "column"),
            eq("table_name", semantic_model.get("table_name", table_name)),
        ] + base_conds
        if semantic_model.get("catalog_name"):
            children_conds.append(eq("catalog_name", semantic_model["catalog_name"]))
        if semantic_model.get("database_name"):
            children_conds.append(eq("database_name", semantic_model["database_name"]))
        if semantic_model.get("schema_name"):
            children_conds.append(eq("schema_name", semantic_model["schema_name"]))

        children = self.storage._search_all(where=And(children_conds)).to_pylist()

        dimensions = []
        measures = []
        identifiers = []

        for child in children:
            child_dict = {
                "name": child.get("name"),
                "description": child.get("description"),
                "expr": child.get("expr") or child.get("name"),
            }

            if child.get("is_dimension"):
                col_type = child.get("column_type")
                if col_type:
                    child_dict["type"] = col_type
                if child.get("is_partition"):
                    child_dict["is_partition"] = True
                if child.get("time_granularity"):
                    child_dict["time_granularity"] = child.get("time_granularity")
                child_dict = {k: v for k, v in child_dict.items() if v is not None and v != ""}
                dimensions.append(child_dict)

            elif child.get("is_measure"):
                if child.get("agg"):
                    child_dict["agg"] = child.get("agg")
                if child.get("create_metric"):
                    child_dict["create_metric"] = True
                if child.get("agg_time_dimension"):
                    child_dict["agg_time_dimension"] = child.get("agg_time_dimension")
                child_dict = {k: v for k, v in child_dict.items() if v is not None and v != ""}
                measures.append(child_dict)

            elif child.get("is_entity_key"):
                col_type = child.get("column_type")
                if col_type:
                    child_dict["type"] = col_type
                if child.get("entity"):
                    child_dict["entity"] = child.get("entity")
                child_dict = {k: v for k, v in child_dict.items() if v is not None and v != ""}
                identifiers.append(child_dict)

        full_result = {
            "catalog_name": semantic_model.get("catalog_name", ""),
            "database_name": semantic_model.get("database_name", ""),
            "schema_name": semantic_model.get("schema_name", ""),
            "table_name": semantic_model.get("table_name", table_name),
            "semantic_model_name": model_name,
            "description": semantic_model.get("description"),
            "yaml_path": semantic_model.get("yaml_path", ""),
            "dimensions": dimensions,
            "measures": measures,
            "identifiers": identifiers,
        }

        if select_fields:
            result = {field: full_result.get(field) for field in select_fields if field in full_result}
        else:
            result = full_result

        return result

    def search_all(self, database_name: str = "", select_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for all table-level semantic model objects."""
        conditions = [eq("kind", "table")] + self._sub_agent_conditions()
        if database_name:
            conditions.append(eq("database_name", database_name))

        where = And(conditions)
        return self.storage._search_all(where=where, select_fields=select_fields).to_pylist()

    def get_size(self) -> int:
        """Get count of table-level semantic model objects (excluding columns)."""
        try:
            return self.storage._count_rows(where=And([eq("kind", "table")] + self._sub_agent_conditions()))
        except Exception:
            return 0

    def store_batch(self, objects: List[Dict[str, Any]]):
        """Store a batch of semantic model objects."""
        self.storage.store_batch(objects)

    def upsert_batch(self, objects: List[Dict[str, Any]]):
        """Upsert a batch of semantic model objects (update if id exists, insert if not)."""
        self.storage.upsert_batch(objects, on_column="id")

    def create_indices(self):
        """Create indices for semantic model storage."""
        self.storage.create_indices()
