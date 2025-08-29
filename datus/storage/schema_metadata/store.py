import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow as pa
from lancedb.rerankers import Reranker

from datus.configuration.agent_config import AgentConfig
from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import TableSchema, TableValue
from datus.storage.base import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel, get_db_embedding_model
from datus.utils.json_utils import json2csv
from datus.utils.loggings import get_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = get_logger(__name__)


class BaseMetadataStorage(BaseEmbeddingStore):
    """
    Base class for metadata storage, include table, view and materialized view(abbreviated as mv).
    properties:
        - db_path: str, database path to store the metadata
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
        db_path: str,
        embedding_model: EmbeddingModel,
        table_name: str,
        vector_source_name: str,
    ):
        super().__init__(
            db_path=db_path,
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
        )
        self.reranker = None

    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n: int = 5,
        table_type: TABLE_TYPE = "table",
        reranker: Optional[Reranker] = None,
    ) -> pa.Table:
        where = _build_where_clause(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
        )
        return self.do_search_similar(query_text, top_n=top_n, where=where, reranker=reranker)

    def do_search_similar(
        self,
        query_text: str,
        top_n: int = 5,
        where: str = "",
        reranker: Optional[Reranker] = None,
    ) -> pa.Table:
        return self.search(
            query_text,
            top_n=top_n,
            where=where,
            reranker=reranker,
        )

    def create_indices(self):
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        # create scalar index
        try:
            self.table.create_scalar_index("database_name", replace=True)
            self.table.create_scalar_index("catalog_name", replace=True)
            self.table.create_scalar_index("schema_name", replace=True)
            self.table.create_scalar_index("table_name", replace=True)
            self.table.create_scalar_index("table_type", replace=True)
        except Exception as e:
            logger.warning(f"Failed to create scalar index for {self.table_name} table: {str(e)}")

        # self.create_vector_index()
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
    """Store and manage schema lineage data in LanceDB."""

    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        """Initialize the schema store.

        Args:
            db_path: Path to the LanceDB database directory
        """
        super().__init__(
            db_path=db_path,
            table_name="schema_metadata",
            embedding_model=embedding_model,
            vector_source_name="definition",
        )
        self.reranker = None
        # self.reranker = CrossEncoderReranker(
        #     model_name="BAAI/bge-reranker-large", device=get_device(), column="schema_text"
        # )

    def _extract_table_name(self, schema_text: str) -> str:
        """Extract table name from CREATE TABLE statement."""
        # Simple extraction - can be enhanced for more complex cases
        words = schema_text.split()
        if len(words) >= 3 and words[0].upper() == "CREATE" and words[1].upper() == "TABLE":
            return words[2].strip("()").strip()
        return ""

    def search_all_schemas(self, database_name: str = "", catalog_name: str = "") -> Set[str]:
        search_result = self._search_all(
            where=_build_where_clause(database_name=database_name, catalog_name=catalog_name),
            select_fields=["schema_name"],
        )
        return {search_result["schema_name"]}

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


class SchemaValueStorage(BaseMetadataStorage):
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        super().__init__(
            db_path=db_path,
            embedding_model=embedding_model,
            table_name="schema_value",
            vector_source_name="sample_rows",
        )
        self.reranker = None
        # self.reranker = CrossEncoderReranker(
        #     model_name="BAAI/bge-reranker-large", device=get_device(), column="sample_rows"
        # )


class SchemaWithValueRAG:
    def __init__(
        self,
        db_path: str,
        # use_rerank: bool = False,
    ):
        self.db_path = db_path

        embedding_model = get_db_embedding_model()
        self.schema_store = SchemaStorage(db_path, embedding_model)
        self.value_store = SchemaValueStorage(db_path, embedding_model)

    def store_batch(self, schemas: List[Dict[str, Any]], values: List[Dict[str, Any]]):
        # Process schemas and values in batches of 500
        # batch_size = 500
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

        logger.debug(f"store {len(schemas)} schemas, {len(final_values)} values")

    def after_init(self):
        """After init the schema and value, create the indices for the tables."""
        self.schema_store.create_indices()
        self.value_store.create_indices()

    def get_schema_size(self):
        return self.schema_store.table_size()

    def get_value_size(self):
        return self.value_store.table_size()

    def search_similar(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        use_rerank: bool = False,
        table_type: TABLE_TYPE = "table",
        top_n: int = 5,
    ) -> Tuple[pa.Table, pa.Table]:
        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
        )
        schema_results = self.schema_store.do_search_similar(
            query_text,
            top_n=top_n,
            where=where,
            reranker=self.schema_store.reranker if use_rerank else None,
        )
        value_results = self.value_store.do_search_similar(
            query_text,
            top_n=top_n,
            where=where,
            reranker=self.value_store.reranker if use_rerank else None,
        )
        return schema_results, value_results

    def search_all_schemas(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "full",
        select_fields: Optional[List[str]] = None,
    ) -> pa.Table:
        """Search all schemas for a given database name.
        Args:
            database_name: The catalog name to search for. If not provided, search all catalogs.
            catalog_name:  The database name to search for. If not provided, search all databases.
            schema_name: The schema name to search for. If not provided, search all schemas.
            table_type: The table type to search for.
            select_fields: The fields to search for. If not provided, search all fields.

        Returns:
            A list of dictionaries containing the schema information.
        """
        return self.schema_store.search_all(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_type=table_type,
            select_fields=select_fields,
        )

    def search_all_value(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_type: TABLE_TYPE = "full"
    ) -> pa.Table:
        """Search all schemas for a given database name.
        :param database_name: The catalog name to search for. If not provided, search all catalogs.
        :param catalog_name:  The database name to search for. If not provided, search all databases.
        :param schema_name: The schema name to search for. If not provided, search all schemas.
        :param table_type: The table type to search for.
        Returns:
            A list of dictionaries containing the schema information.
        """
        return self.value_store.search_all(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_type=table_type
        )

    def search_tables(self, database_name: str, tables: list[str]) -> Tuple[List[TableSchema], List[TableValue]]:
        """
        Search schemas and values for given table names.
        """
        # Ensure tables are ready before direct table access
        self.schema_store._ensure_table_ready()
        self.value_store._ensure_table_ready()

        # Parse table names and build where clause
        conditions = []
        for full_table in tables:
            parts = full_table.split(".")
            if len(parts) == 3:
                # Format: database_name.schema_name.table_name
                conditions.append(
                    f"(database_name='{parts[0]}' AND schema_name='{parts[1]}' " f"AND table_name='{parts[2]}')"
                )
            elif len(parts) == 2:
                # Format: database_name.table_name(Maybe need fix for other dialects)
                conditions.append(f"(database_name='{parts[0]}' AND table_name='{parts[1]}')")
            else:
                conditions.append(f"(database_name='{database_name}' AND table_name='{full_table}')")
        where_clause = " OR ".join(conditions)
        # Search schemas
        schema_results = self.schema_store.table.search().where(where_clause).limit(len(tables)).to_arrow()
        schemas_result = TableSchema.from_arrow(schema_results)

        value_results = self.value_store.table.search().where(where_clause).limit(len(tables)).to_arrow()
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
        # Ensure tables are ready before deletion
        self.schema_store._ensure_table_ready()
        self.value_store._ensure_table_ready()

        where = _build_where_clause(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            table_type=table_type,
        )
        self.schema_store.table.delete(where)
        self.value_store.table.delete(where)


def _build_where_clause(
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    table_name: str = "",
    table_type: TABLE_TYPE = "table",
) -> str:
    where = ""
    if catalog_name:
        where += f"catalog_name='{catalog_name}'"
    if database_name:
        where += f" AND database_name='{database_name}'" if where else f"database_name='{database_name}'"
    if schema_name:
        where += f" AND schema_name='{schema_name}'" if where else f"schema_name='{schema_name}'"
    if table_name:
        where += f" AND table_name='{table_name}'" if where else f"table_name='{table_name}'"
    if table_type and table_type != "full":
        where += f" AND table_type='{table_type}'" if where else f"table_type='{table_type}'"
    return where


def rag_by_configuration(agent_config: AgentConfig):
    return SchemaWithValueRAG(agent_config.rag_storage_path())
