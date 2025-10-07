import logging
from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.storage.embedding_models import get_metric_embedding_model
from datus.storage.lancedb_conditions import And, build_where, eq, in_
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
        self.table.create_scalar_index("table_name", replace=True)
        self.table.create_scalar_index("domain", replace=True)
        self.table.create_scalar_index("layer1", replace=True)
        self.table.create_scalar_index("layer2", replace=True)
        self.create_fts_index(["semantic_model_name", "semantic_model_desc", "identifiers", "dimensions", "measures"])

    def search_all(self, database_name: str = "", select_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search all schemas for a given database name."""

        search_result = self._search_all(
            where=None if not database_name else eq("database_name", database_name),
            select_fields=select_fields,
        )
        return search_result.to_pylist()

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
                    pa.field("name", pa.string()),
                    pa.field("llm_text", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="llm_text",
        )
        self.reranker = None

    def create_indices(self):
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        self.table.create_scalar_index("id", replace=True)
        self.table.create_scalar_index("semantic_model_name", replace=True)
        self.create_fts_index(["name"])

        # Add independent indices for domain/layer queries
        self.table.create_scalar_index("domain", replace=True)
        self.table.create_scalar_index("layer1", replace=True)
        self.table.create_scalar_index("layer2", replace=True)

        # Keep combined index for uniqueness constraint
        self.table.create_scalar_index("domain_layer1_layer2", replace=True)

        self.create_fts_index(["name", "description", "constraint", "sql_query"])

    def search_all(
        self, semantic_model_name: str = "", select_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search all schemas for a given database name."""
        # Ensure table is ready before direct table access
        self._ensure_table_ready()

        search_result = self._search_all(
            where=None if not semantic_model_name else eq("semantic_model_name", semantic_model_name),
            select_fields=select_fields,
        )
        return search_result.to_pylist()


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
        self, database_name: str, select_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        return self.semantic_model_storage.search_all(database_name, select_fields=select_fields)

    def search_all_metrics(
        self, semantic_model_name: str = "", select_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        return self.metric_storage.search_all(semantic_model_name, select_fields=select_fields)

    def after_init(self):
        self.semantic_model_storage.create_indices()
        self.metric_storage.create_indices()

    def get_semantic_model_size(self):
        return self.semantic_model_storage.table_size()

    def get_metrics_size(self):
        return self.metric_storage.table_size()

    def get_domains(self) -> List[str]:
        """Get all unique domains from metrics.

        Returns:
            List of unique domain names
        """
        self.metric_storage._ensure_table_ready()

        MAX_DOMAINS = 1000
        search_result = self.metric_storage.table.search().select(["domain"]).limit(MAX_DOMAINS).to_list()

        if len(search_result) >= MAX_DOMAINS:
            logger.warning(
                f"Retrieved {MAX_DOMAINS} domain records (may be truncated). "
                "Consider using filters to reduce result set."
            )

        domains = list(set(result["domain"] for result in search_result if result.get("domain")))
        logger.debug(f"Found {len(domains)} unique domains from {len(search_result)} records")

        return sorted(domains)

    def get_layers_by_domain(self, domain: str = "") -> List[tuple]:
        """Get unique (layer1, layer2) combinations from metrics.

        Args:
            domain: Domain name to filter. Leave empty to get all layers.

        Returns:
            List of (layer1, layer2) tuples. Maximum 1000 combinations.
        """
        MAX_LAYERS = 1000

        self.metric_storage._ensure_table_ready()

        where_clause = build_where(eq("domain", domain)) if domain else None
        search_result = self.metric_storage.table.search()
        if where_clause:
            search_result = search_result.where(where_clause)

        results = search_result.select(["layer1", "layer2"]).limit(MAX_LAYERS).to_list()

        if len(results) >= MAX_LAYERS:
            logger.warning(
                f"Retrieved {MAX_LAYERS} layer records (may be truncated). "
                f"Consider filtering by domain to reduce result set."
            )

        unique_layers = set((r["layer1"], r["layer2"]) for r in results if r.get("layer1") and r.get("layer2"))
        logger.debug(f"Found {len(unique_layers)} unique layers from {len(results)} records")

        return list(unique_layers)

    def get_metrics_names(self, domain: str, layer1: str, layer2: str) -> List[Dict[str, str]]:
        """Get metric names with descriptions in a specific layer.

        Args:
            domain: Domain name
            layer1: Primary layer name
            layer2: Secondary layer name

        Returns:
            List of dicts with 'name' and 'description' fields
        """
        MAX_METRICS = 1000

        conditions = [eq("domain", domain), eq("layer1", layer1), eq("layer2", layer2)]
        query_result = self.metric_storage._search_all(
            And(conditions),
            select_fields=["name", "description"],
        )

        if query_result is None or query_result.num_rows == 0:
            logger.debug(f"No metrics found for domain={domain}, layer1={layer1}, layer2={layer2}")
            return []

        if query_result.num_rows >= MAX_METRICS:
            logger.warning(
                f"Retrieved {query_result.num_rows} metrics (may be truncated at {MAX_METRICS}). "
                "Result set is very large."
            )

        result_list = query_result.to_pylist()
        logger.debug(f"Found {len(result_list)} metrics in {domain}/{layer1}/{layer2}")

        return result_list

    def search_hybrid_metrics(
        self,
        query_text: str,
        domain: str = "",
        layer1: str = "",
        layer2: str = "",
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n: int = 5,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        semantic_conditions = []
        if catalog_name:
            semantic_conditions.append(eq("catalog_name", catalog_name))
        if database_name:
            semantic_conditions.append(eq("database_name", database_name))
        if schema_name:
            semantic_conditions.append(eq("schema_name", schema_name))

        semantic_condition = And(semantic_conditions) if semantic_conditions else None
        semantic_where_clause = build_where(semantic_condition) if semantic_condition else None
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
            select_fields=["llm_text"],
            top_n=top_n,
            where=metric_condition,
        )

        if metric_search_results is None or metric_search_results.num_rows == 0:
            logger.info("Metric search returned no results")
            return []

        try:
            metric_result = metric_search_results.select(["llm_text"]).to_pylist()
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
                "llm_text",
            ],
        )
        return search_result.to_pylist()

    def get_metrics(
        self,
        domain: str = "",
        layer1: str = "",
        layer2: str = "",
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
        self.metric_storage.update(where, update_payload, unique_filter=unique_filter)

    def update_semantic_model(self, old_values: Dict[str, Any], update_values: Dict[str, Any]):
        where = And(
            [
                eq("catalog_name", old_values["catalog_name"]),
                eq("database_name", old_values["database_name"]),
                eq("schema_name", old_values["schema_name"]),
                eq("table_name", old_values["table_name"]),
                eq("semantic_model_name", old_values["semantic_model_name"]),
            ]
        )
        self.semantic_model_storage.update(where, update_values, unique_filter=None)


def rag_by_configuration(agent_config: AgentConfig):
    return SemanticMetricsRAG(agent_config.rag_storage_path())
