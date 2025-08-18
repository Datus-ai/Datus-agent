import logging
from typing import Any, Dict, List

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.schemas.search_metrics_node_models import SearchMetricsInput
from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.storage.embedding_models import get_metric_embedding_model

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
        self.create_fts_index(["semantic_model_name", "semantic_model_desc", "identifiers", "dimensions", "measures"])

    def search_all(self, database_name: str = "") -> List[Dict[str, Any]]:
        """Search all schemas for a given database name."""

        search_result = self._search_all(
            where="" if not database_name else f"database_name='{database_name}'",
            select_fields=["id", "catalog_database_schema", "semantic_model_name"],
        )
        return [
            {
                "id": search_result["id"][i],
                "catalog_database_schema": search_result["catalog_database_schema"][i],
                "semantic_model_name": search_result["semantic_model_name"][i],
            }
            for i in range(search_result.num_rows)
        ]

    def filter_by_id(self, id: str) -> List[Dict[str, Any]]:
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        search_result = self.table.search().where(f"id = '{id}'").limit(100).to_list()
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
                    pa.field("metric_name", pa.string()),
                    pa.field("metric_value", pa.string()),
                    pa.field("metric_type", pa.string()),
                    pa.field("metric_sql_query", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="metric_value",
        )
        self.reranker = None

    def create_indices(self):
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        self.table.create_scalar_index("id", replace=True)
        self.table.create_scalar_index("semantic_model_name", replace=True)
        self.table.create_scalar_index("domain_layer1_layer2", replace=True)
        self.create_fts_index(["metric_name", "metric_value", "metric_sql_query"])

    def search_all(self, semantic_model_name: str) -> List[Dict[str, Any]]:
        """Search all schemas for a given database name."""
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        search_result = self._search_all(
            where="" if not semantic_model_name else f"semantic_model_name='{semantic_model_name}'",
            select_fields=["id", "semantic_model_name", "metric_name"],
        )
        return [
            {
                "id": search_result["id"][i],
                "semantic_model_name": search_result["semantic_model_name"][i],
                "metric_name": search_result["metric_name"][i],
            }
            for i in range(search_result.num_rows)
        ]


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

    def search_all_semantic_models(self, database_name: str) -> List[Dict[str, Any]]:
        return self.semantic_model_storage.search_all(database_name)

    def search_all_metrics(self, database_name: str) -> List[Dict[str, Any]]:
        return self.metric_storage.search_all(database_name)

    def after_init(self):
        self.semantic_model_storage.create_indices()
        self.metric_storage.create_indices()

    def get_semantic_model_size(self):
        return self.semantic_model_storage.table_size()

    def get_metrics_size(self):
        return self.metric_storage.table_size()

    def search_hybrid_metrics(
        self, input_param: SearchMetricsInput, top_n: int = 5, use_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        query_text = input_param.input_text
        semantic_full_name: str = qualify_name(
            [
                input_param.sql_task.catalog_name,
                input_param.sql_task.database_name,
                input_param.sql_task.schema_name,
            ]
        )

        semantic_where = f"catalog_database_schema = '{semantic_full_name}'"
        if "%" in semantic_where:
            semantic_where = f"catalog_database_schema like '{semantic_full_name}'"
        logger.info(f"start to search semantic, semantic_where: {semantic_where}, query_text: {query_text}")
        semantic_search_results = self.semantic_model_storage.search(
            query_text,
            top_n=top_n,
            where=semantic_where,
        )

        metric_full_name: str = qualify_name(
            [
                input_param.sql_task.domain,
                input_param.sql_task.layer1,
                input_param.sql_task.layer2,
            ],
        )
        metric_where = f"domain_layer1_layer2 = '{metric_full_name}'"
        if "%" in metric_where:
            metric_where = f"domain_layer1_layer2 like '{metric_full_name}'"
        logger.info(f"start to search metrics, metric_where: {metric_where}, query_text: {query_text}")
        metric_search_results = self.metric_storage.search(query_text, top_n=top_n, where=metric_where)

        # get the intersection result from semantic_results & metric_results
        metric_result = []
        if semantic_search_results and metric_search_results:
            try:
                semantic_names_set = set(
                    [result["semantic_model_name"] for result in semantic_search_results.to_pylist()]
                )
                for result in metric_search_results.to_pylist():
                    if result["semantic_model_name"] in semantic_names_set:
                        filtered_result = {
                            k: v
                            for k, v in result.items()
                            if k in ["metric_name", "metric_value", "metric_type", "metric_sql_query"]
                        }
                        metric_result.append(filtered_result)
            except Exception as e:
                logger.warning(f"Failed to get the intersection result, exception: {str(e)}")

        logger.info(f"Got the metrics result, size: {len(metric_result)}, query_text: {query_text}")
        return metric_result


def rag_by_configuration(agent_config: AgentConfig):
    return SemanticMetricsRAG(agent_config.rag_storage_path())
