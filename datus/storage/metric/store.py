import logging
from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.storage.embedding_models import get_metric_embedding_model
from datus.storage.lancedb_conditions import And, build_where, eq, in_, like
from datus.utils.exceptions import DatusException, ErrorCode

logger = logging.getLogger(__file__)


class SemanticModelStorage(BaseEmbeddingStore):
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        """Initialize the schema store.

        Args:
            db_path: Path to the LanceDB database directory
        """
        super().__init__(
            db_path=db_path,
            table_name="semantic_model",
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    pa.field("table_name", pa.string()),
                    pa.field("catalog_database_schema", pa.string()),
                    pa.field("domain", pa.string()),
                    pa.field("layer1", pa.string()),
                    pa.field("layer2", pa.string()),
                    pa.field("semantic_file_path", pa.string()),
                    pa.field("semantic_model_name", pa.string()),
                    pa.field("semantic_model_desc", pa.string()),
                    pa.field("identifiers", pa.string()),
                    pa.field("dimensions", pa.string()),
                    pa.field("measures", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="dimensions",
        )
        self.reranker = None

    def create_indices(self):
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        self.table.create_scalar_index("id", replace=True)
        self.table.create_scalar_index("catalog_name", replace=True)
        self.table.create_scalar_index("database_name", replace=True)
        self.table.create_scalar_index("schema_name", replace=True)
        self.table.create_scalar_index("catalog_database_schema", replace=True)
        self.table.create_scalar_index("table_name", replace=True)
        self.table.create_scalar_index("domain", replace=True)
        self.table.create_scalar_index("layer1", replace=True)
        self.table.create_scalar_index("layer2", replace=True)
        self.create_fts_index(["semantic_model_name", "semantic_model_desc", "identifiers", "dimensions", "measures"])

    def search_all(self, database_name: str = "", selected_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search all schemas for a given database name."""

        search_result = self._search_all(
            where=None if not database_name else eq("database_name", database_name),
            select_fields=selected_fields,
        )
        if not selected_fields:
            return search_result.to_pylist()
        result = []
        for i in range(search_result.num_rows):
            d = {}
            for k in selected_fields:
                d[k] = search_result[k][i]
        return result

    def filter_by_id(self, id: str) -> List[Dict[str, Any]]:
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        where_clause = build_where(eq("id", id))
        search_result = self.table.search().where(where_clause).limit(100).to_list()
        return search_result


class MetricStorage(BaseEmbeddingStore):
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        super().__init__(
            db_path=db_path,
            table_name="metrics",
            embedding_model=embedding_model,
            schema=pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("semantic_model_name", pa.string()),
                    pa.field("domain", pa.string()),
                    pa.field("layer1", pa.string()),
                    pa.field("layer2", pa.string()),
                    pa.field("domain_layer1_layer2", pa.string()),
                    pa.field("name", pa.string()),
                    pa.field("description", pa.string()),
                    pa.field("constraint", pa.string()),
                    pa.field("sql_query", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="description",
        )
        self.reranker = None

    def create_indices(self):
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        self.table.create_scalar_index("id", replace=True)
        self.table.create_scalar_index("semantic_model_name", replace=True)
        self.table.create_scalar_index("domain_layer1_layer2", replace=True)
        self.create_fts_index(["name", "description", "constraint", "sql_query"])

    def search_all(self, semantic_model_name: str = "", select_fields: Optional[List[str]] = None) -> pa.Table:
        """Search all schemas for a given database name."""
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        return self._search_all(
            where=None if not semantic_model_name else eq("semantic_model_name", semantic_model_name),
            select_fields=select_fields,
        )


def qualify_name(input_names: List, delimiter: str = "_") -> str:
    names = []
    for name in input_names:
        if not name:
            names.append("%")
        else:
            names.append(name)
    return delimiter.join(names)


class SemanticMetricsRAG:
    def __init__(self, db_path: str):
        self.db_path = db_path
        embedding_model = get_metric_embedding_model()
        self.semantic_model_storage = SemanticModelStorage(db_path, embedding_model)
        self.metric_storage = MetricStorage(db_path, embedding_model)

    def store_batch(self, semantic_models: List[Dict[str, Any]], metrics: List[Dict[str, Any]]):
        logger.info(f"store semantic models: {semantic_models}")
        logger.info(f"store metrics: {metrics}")
        self.semantic_model_storage.store_batch(semantic_models)
        self.metric_storage.store_batch(metrics)

    def search_all_semantic_models(
        self, database_name: str, selected_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        return self.semantic_model_storage.search_all(database_name, selected_fields=selected_fields)

    def search_all_metrics(
        self, semantic_model_name: str = "", select_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        search_result = self.metric_storage.search_all(semantic_model_name, select_fields=select_fields)
        if select_fields:
            # Only return selected fields
            return [
                {field: search_result[field][i].as_py() for field in select_fields}
                for i in range(search_result.num_rows)
            ]
        else:
            # Return all fields
            return [
                {field: search_result[field][i].as_py() for field in search_result.column_names}
                for i in range(search_result.num_rows)
            ]

    def after_init(self):
        self.semantic_model_storage.create_indices()
        self.metric_storage.create_indices()

    def get_semantic_model_size(self):
        return self.semantic_model_storage.table_size()

    def get_metrics_size(self):
        return self.metric_storage.table_size()

    def search_hybrid_metrics(
        self,
        query_text: str,
        domain: str,
        layer1: str,
        layer2: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n: int = 5,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        semantic_full_name: str = qualify_name(
            [
                catalog_name,
                database_name,
                schema_name,
            ]
        )

        semantic_condition = (
            like("catalog_database_schema", semantic_full_name)
            if "%" in semantic_full_name
            else eq("catalog_database_schema", semantic_full_name)
        )
        semantic_where_clause = build_where(semantic_condition)
        logger.info(f"start to search semantic, semantic_where: {semantic_where_clause}, query_text: {query_text}")
        semantic_search_results = self.semantic_model_storage.search(
            query_text,
            select_fields=["semantic_model_name"],
            top_n=top_n,
            where=semantic_condition,
        )

        if semantic_search_results is None or semantic_search_results.num_rows == 0:
            logger.info("No semantic matches found; skipping metric search")
            return []

        semantic_names = [name for name in semantic_search_results["semantic_model_name"].to_pylist() if name]
        if not semantic_names:
            logger.info("Semantic search returned no model names; skipping metric search")
            return []
        conditions = [in_("semantic_model_name", semantic_names)]
        if domain:
            conditions.append(eq("domain", domain))
        if layer1:
            conditions.append(eq("layer1", layer1))
        if layer2:
            conditions.append(eq("layer2", layer2))

        metric_condition = And(conditions)
        metric_where_clause = build_where(metric_condition)
        logger.info(f"start to search metrics, metric_where: {metric_where_clause}, query_text: {query_text}")
        metric_search_results = self.metric_storage.search(
            query_txt=query_text,
            select_fields=["name", "description", "constraint", "sql_query", "semantic_model_name"],
            top_n=top_n,
            where=metric_condition,
        )

        if metric_search_results is None or metric_search_results.num_rows == 0:
            logger.info("Metric search returned no results")
            return []

        try:
            metric_result = metric_search_results.select(
                ["name", "description", "constraint", "sql_query", "semantic_model_name"]
            ).to_pylist()
        except Exception as e:
            logger.warning(f"Failed to extract metric results, exception: {str(e)}")
            return []

        logger.info(f"Got the metrics result, size: {len(metric_result)}, query_text: {query_text}")
        return metric_result

    def get_metrics_detail(self, domain: str, layer1: str, layer2: str, name: str) -> List[Dict[str, Any]]:
        metric_condition = And(
            [
                eq("domain", domain),
                eq("layer1", layer1),
                eq("layer2", layer2),
                eq("name", name),
            ]
        )

        search_result = self.metric_storage._search_all(
            where=metric_condition,
            select_fields=[
                "domain",
                "layer1",
                "layer2",
                "name",
                "semantic_model_name",
                "description",
                "constraint",
                "sql_query",
            ],
        )
        return search_result.to_pylist()

    def get_metrics(
        self,
        domain: str = "default",
        layer1: str = "default",
        layer2: str = "default",
        semantic_model_name: str = "",
        selected_fields: Optional[List[str]] = None,
        return_distance: bool = False,
    ) -> List[Dict[str, Any]]:
        conditions = [eq("domain", domain), eq("layer1", layer1), eq("layer2", layer2)]
        if semantic_model_name:
            conditions.append(eq("semantic_model_name", semantic_model_name))
        query_result = self.metric_storage._search_all(
            And(conditions),
            select_fields=selected_fields,
        )
        if return_distance:
            return query_result.to_pylist()
        else:
            columns = query_result.column_names
            if "_distance" in columns:
                return query_result.remove_column(columns.index("_distance")).to_pylist()
            return query_result.to_pylist()

    def update_metrics(self, old_values: Dict[str, Any], update_values: Dict[str, Any]):
        """
        Currently, only two update scenarios are supported:
            - Update domain, layer 1, layer 2, and name
            - Update detail fields
        """
        if "name" in update_values:
            unique_filter = And(
                [
                    eq("domain", update_values.get("domain", old_values.get("domain"))),
                    eq("layer1", update_values.get("layer1", old_values.get("layer1"))),
                    eq("layer2", update_values.get("layer2", old_values.get("layer2"))),
                    eq("name", update_values["name"]),
                ]
            )
        else:
            unique_filter = None
        where_conditions = []
        for k in ("domain", "layer1", "layer2", "name"):
            if k in old_values:
                where_conditions.append(eq(k, old_values[k]))

        where = And(where_conditions)
        if not where_conditions:
            raise DatusException(
                ErrorCode.STORAGE_TABLE_OPERATION_FAILED,
                message_args={
                    "operation": "update",
                    "table_name": self.metric_storage.table_name,
                    "error_message": "Missing WHERE for metrics update",
                },
            )
        update_payload = dict(update_values)
        domain_value = update_payload.get("domain", old_values.get("domain"))
        layer1_value = update_payload.get("layer1", old_values.get("layer1"))
        layer2_value = update_payload.get("layer2", old_values.get("layer2"))
        if domain_value and layer1_value and layer2_value:
            update_payload["domain_layer1_layer2"] = qualify_name([domain_value, layer1_value, layer2_value])
        self.metric_storage.update(where, update_payload, unique_filter=unique_filter)

    def update_semantic_model(self, old_values: Dict[str, Any], update_values: Dict[str, Any]):
        unique_filter = And(
            [
                eq("catalog_name", update_values.get("catalog_name", old_values["catalog_name"])),
                eq("database_name", update_values.get("database_name", old_values["database_name"])),
                eq("schema_name", update_values.get("schema_name", old_values["schema_name"])),
                eq("table_name", update_values.get("table_name", old_values["table_name"])),
                eq("semantic_model_name", update_values["semantic_model_name"]),
            ]
        )
        where = And(
            [
                eq("catalog_name", old_values["catalog_name"]),
                eq("database_name", old_values["database_name"]),
                eq("schema_name", old_values["schema_name"]),
                eq("table_name", old_values["table_name"]),
                eq("semantic_model_name", old_values["semantic_model_name"]),
            ]
        )
        self.semantic_model_storage.update(where, update_values, unique_filter=unique_filter)


def rag_by_configuration(agent_config: AgentConfig):
    return SemanticMetricsRAG(agent_config.rag_storage_path())
