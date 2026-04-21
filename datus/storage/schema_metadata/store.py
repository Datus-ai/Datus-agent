# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow as pa
from datus_storage_base.conditions import Node, and_, eq, or_

from datus.configuration.agent_config import AgentConfig
from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import TableSchema, TableValue
from datus.storage.base import BaseEmbeddingStore, WhereExpr
from datus.storage.embedding_models import EmbeddingModel
from datus.tools.db_tools import connector_registry
from datus.utils.constants import DBType
from datus.utils.json_utils import json2csv
from datus.utils.loggings import get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


class BaseMetadataStorage(BaseEmbeddingStore):
    """
    Base class for metadata storage, include table, view and materialized view(abbreviated as mv).
    properties:
        - embedding_model: EmbeddingModel, embedding model to embed the metadata
        - table_name: str, table name to store the metadata
        - vector_source_name: str, vector source name, required, should define in subclass
        - reranker: Reranker, reranker, optional

    schema properties:
        - identifier: str, unique identifier for the metadata, spliced by catalog_name, database_name, schema_name,
        table_name, table_type
        - catalog_name: str, catalog name, optional
        - database_name: str, database name, optional
        - schema_name: str, schema name, optional
        - table_name: str, table name, required
        - table_type: str, table type, choices: table, view, mv
        - vector_source_name: str, vector source name, required
        - vector: list[float], vector, required
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        table_name: str,
        vector_source_name: str,
        **kwargs,
    ):
        super().__init__(
            table_name=table_name,
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    pa.field("identifier", pa.string()),
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    pa.field("table_name", pa.string()),
                    pa.field("table_type", pa.string()),
                    pa.field(vector_source_name, pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name=vector_source_name,
            **kwargs,
        )

    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n: int = 5,
        table_type: TABLE_TYPE = "table",
        query_type: str = "vector",
    ) -> pa.Table:
        where = _build_where_clause(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
        )
        return self.do_search_similar(query_text, top_n=top_n, where=where, query_type=query_type)

    def do_search_similar(
        self,
        query_text: str,
        top_n: int = 5,
        where: WhereExpr = None,
        query_type: str = "vector",
    ) -> pa.Table:
        return self.search(
            query_text,
            top_n=top_n,
            where=where,
            query_type=query_type,
        )

    def create_indices(self):
        self._ensure_table_ready()

        self._create_scalar_index("database_name")
        self._create_scalar_index("catalog_name")
        self._create_scalar_index("schema_name")
        self._create_scalar_index("table_name")
        self._create_scalar_index("table_type")

        self.create_fts_index(["database_name", "schema_name", "table_name", self.vector_source_name])

    def search_all(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "full",
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """Search all schemas for a given database name."""
        # Ensure table is ready before searching
        self._ensure_table_ready()

        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        return self._search_all(where=where, select_fields=select_fields)


class SchemaStorage(BaseMetadataStorage):
    """Store and manage schema lineage data."""

    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        """Initialize the schema store."""
        super().__init__(
            table_name="schema_metadata",
            embedding_model=embedding_model,
            vector_source_name="definition",
            **kwargs,
        )

    def _extract_table_name(self, schema_text: str) -> str:
        """Extract table name from CREATE TABLE statement."""
        words = schema_text.split()
        if len(words) < 3 or words[0].upper() != "CREATE" or words[1].upper() != "TABLE":
            return ""
        idx = 2
        # Skip IF NOT EXISTS
        if idx + 2 < len(words) and words[idx].upper() == "IF" and words[idx + 1].upper() == "NOT":
            idx += 3  # skip IF NOT EXISTS
        if idx >= len(words):
            return ""
        name = words[idx]
        # Handle table name with no space before paren, e.g. "mytable(id INT)"
        paren_pos = name.find("(")
        if paren_pos > 0:
            name = name[:paren_pos]
        return name.strip("()").strip()

    def search_all_schemas(self, database_name: str = "", catalog_name: str = "") -> Set[str]:
        search_result = self._search_all(
            where=_build_where_clause(database_name=database_name, catalog_name=catalog_name),
            select_fields=["schema_name"],
        )
        return set(search_result["schema_name"].to_pylist())

    def search_top_tables_by_every_schema(
        self,
        query_text: str,
        database_name: str = "",
        catalog_name: str = "",
        all_schemas: Optional[Set[str]] = None,
        top_n: int = 20,
    ) -> pa.Table:
        if all_schemas is None:
            all_schemas = self.search_all_schemas(catalog_name=catalog_name, database_name=database_name)
        result = []
        for schema in all_schemas:
            result.append(
                self.search_similar(
                    query_text=query_text,
                    database_name=database_name,
                    catalog_name=catalog_name,
                    schema_name=schema,
                    top_n=top_n,
                )
            )
        return pa.concat_tables(result, promote_options="default")

    def get_schema(
        self,
        table_name: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        extra_where: WhereExpr = None,
    ) -> pa.Table:
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type="full",
        )
        table_condition = eq("table_name", table_name)
        if where:
            where_condition = and_(where, table_condition)
        else:
            where_condition = table_condition
        if extra_where:
            where_condition = and_(where_condition, extra_where)
        self._ensure_table_ready()
        return self.table.search_all(
            where=where_condition,
            select_fields=["catalog_name", "database_name", "schema_name", "table_name", "table_type", "definition"],
        )


class SchemaValueStorage(BaseMetadataStorage):
    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        super().__init__(
            embedding_model=embedding_model,
            table_name="schema_value",
            vector_source_name="sample_rows",
            **kwargs,
        )


class SchemaWithValueRAG:
    """RAG interface for schema metadata operations.

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
        self.schema_store = get_storage(SchemaStorage, "database", project=agent_config.project_name)
        self.value_store = get_storage(SchemaValueStorage, "database", project=agent_config.project_name)
        self._sub_agent_filter = _build_sub_agent_filter(agent_config, sub_agent_name, self.schema_store, "tables")

    def _sub_agent_conditions(self) -> list:
        """Build sub-agent filter conditions (datasource_id handled by backend)."""
        conditions = []
        if self._sub_agent_filter:
            conditions.append(self._sub_agent_filter)
        return conditions

    def _add_sub_agent_filter(self, where: WhereExpr) -> WhereExpr:
        """Add sub-agent filter to existing where clause."""
        conditions = self._sub_agent_conditions()
        if not conditions:
            return where
        sub_agent_filter = conditions[0] if len(conditions) == 1 else and_(*conditions)
        if where is None:
            return sub_agent_filter
        return and_(where, sub_agent_filter)

    def truncate(self) -> None:
        """Delete all schema metadata for this datasource."""
        self.schema_store.truncate_scoped()
        self.value_store.truncate_scoped()

    def store_batch(self, schemas: List[Dict[str, Any]], values: List[Dict[str, Any]]):
        if schemas:
            self.schema_store.store_batch(schemas)

        if len(values) == 0:
            return

        final_values = []
        for item in values:
            if "sample_rows" not in item or not item["sample_rows"]:
                continue
            sample_rows = item["sample_rows"]
            if isinstance(sample_rows, list):
                sample_rows = json2csv(sample_rows)
            item["sample_rows"] = sample_rows
            final_values.append(item)
        self.value_store.store_batch(final_values)

        logger.debug(f"Batch stored {len(schemas)} schemas, {len(final_values)} values")

    def after_init(self):
        """After init the schema and value, create the indices for the tables."""
        self.schema_store.create_indices()
        self.value_store.create_indices()

    def get_schema_size(self):
        return self.schema_store._count_rows(where=self._add_sub_agent_filter(None))

    def get_value_size(self):
        return self.value_store._count_rows(where=self._add_sub_agent_filter(None))

    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        query_type: str = "vector",
        table_type: TABLE_TYPE = "table",
        top_n: int = 5,
    ) -> Tuple[pa.Table, pa.Table]:
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        where = self._add_sub_agent_filter(where)
        schema_results = self.schema_store.do_search_similar(
            query_text,
            top_n=top_n,
            where=where,
            query_type=query_type,
        )
        value_results = self.value_store.do_search_similar(
            query_text,
            top_n=top_n,
            where=where,
            query_type=query_type,
        )
        return schema_results, value_results

    def get_schema(
        self, table_name: str, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> pa.Table:
        """Get schema for a specific table with datasource_id filtering."""
        return self.schema_store.get_schema(
            table_name=table_name,
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            extra_where=self._add_sub_agent_filter(None),
        )

    def search_all_schemas(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "full",
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """Search all schemas for a given database name."""
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        where = self._add_sub_agent_filter(where)
        return self.schema_store._search_all(where=where, select_fields=select_fields)

    def search_all_value(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_type: TABLE_TYPE = "full"
    ) -> pa.Table:
        """Search all values for a given database name."""
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        where = self._add_sub_agent_filter(where)
        return self.value_store._search_all(where=where)

    def search_tables(
        self,
        tables: list[str],
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        dialect: str = DBType.SQLITE,
    ) -> Tuple[List[TableSchema], List[TableValue]]:
        """
        Search schemas and values for given table names.
        """
        # Ensure tables are ready before direct table access
        self.schema_store._ensure_table_ready()
        self.value_store._ensure_table_ready()

        # Parse table names and build where clause
        table_conditions = []
        for full_table in tables:
            parts = full_table.split(".")
            table_name = parts[-1]
            if len(parts) == 4:
                cat, db, sch = parts[0], parts[1], parts[2]
                table_conditions.append(
                    _build_where_clause(
                        table_name=table_name,
                        catalog_name=cat,
                        database_name=db,
                        schema_name=sch,
                        table_type="full",
                    )
                )
            elif len(parts) == 3:
                # Format depends on dialect capabilities:
                # - catalog + no schema (e.g., StarRocks): catalog.database.table
                # - with schema (e.g., PostgreSQL, Snowflake): database.schema.table
                if connector_registry.support_catalog(dialect) and not connector_registry.support_schema(dialect):
                    cat, db, sch = parts[0], parts[1], ""
                else:
                    cat, db, sch = catalog_name, parts[0], parts[1]

                table_conditions.append(
                    _build_where_clause(
                        table_name=table_name,
                        catalog_name=cat,
                        database_name=db,
                        schema_name=sch,
                        table_type="full",
                    )
                )
            elif len(parts) == 2:
                # No schema layer: part[0] is database_name
                # Has schema layer: part[0] is schema_name
                if not connector_registry.support_schema(dialect):
                    cat, db, sch = catalog_name, parts[0], ""
                else:
                    cat, db, sch = catalog_name, database_name, parts[0]

                table_conditions.append(
                    _build_where_clause(
                        table_name=table_name,
                        catalog_name=cat,
                        database_name=db,
                        schema_name=sch,
                        table_type="full",
                    )
                )
            else:
                table_conditions.append(
                    _build_where_clause(
                        table_name=table_name,
                        catalog_name=catalog_name,
                        database_name=database_name,
                        schema_name=schema_name,
                        table_type="full",
                    )
                )

        combined_condition = None
        if table_conditions:
            combined_condition = table_conditions[0] if len(table_conditions) == 1 else or_(*table_conditions)
        else:
            combined_condition = None

        # Apply datasource_id + sub-agent scope filter
        schema_condition = self._add_sub_agent_filter(combined_condition)
        value_condition = self._add_sub_agent_filter(combined_condition)

        schema_fields = [
            "identifier",
            "catalog_name",
            "database_name",
            "schema_name",
            "table_name",
            "table_type",
            "definition",
        ]
        value_fields = [
            "identifier",
            "catalog_name",
            "database_name",
            "schema_name",
            "table_name",
            "table_type",
            "sample_rows",
        ]

        schema_results = self.schema_store.query_with_filter(
            where=schema_condition,
            select_fields=schema_fields,
        )
        schemas_result = TableSchema.from_arrow(schema_results)

        value_results = self.value_store.query_with_filter(
            where=value_condition,
            select_fields=value_fields,
        )
        values_result = TableValue.from_arrow(value_results)

        return schemas_result, values_result

    def remove_data(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        table_type: TABLE_TYPE = "table",
    ):
        where_condition = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            table_type=table_type,
        )
        if where_condition:
            self.schema_store._delete_rows(where_condition)
            self.value_store._delete_rows(where_condition)


def _build_where_clause(
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    table_name: str = "",
    table_type: TABLE_TYPE = "table",
) -> Optional[Node]:
    conditions = []
    if catalog_name:
        conditions.append(eq("catalog_name", catalog_name))
    if database_name:
        conditions.append(eq("database_name", database_name))
    if schema_name:
        conditions.append(eq("schema_name", schema_name))
    if table_name:
        conditions.append(eq("table_name", table_name))
    if table_type and table_type != "full":
        conditions.append(eq("table_type", table_type))

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return and_(*conditions)
