# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.storage.lancedb_conditions import And, WhereExpr, and_, build_where, eq, in_

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
    def __init__(self, agent_config: AgentConfig, sub_agent_name: Optional[str] = None):
        from datus.storage.cache import get_storage_cache_instance

        self.semantic_model_storage: SemanticModelStorage = get_storage_cache_instance(agent_config).semantic_storage(
            sub_agent_name
        )
        self.metric_storage: MetricStorage = get_storage_cache_instance(agent_config).metrics_storage(sub_agent_name)

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

    def search_metrics(
        self, query_text: str, domain: str = "", layer1: str = "", layer2: str = "", top_n: int = 5
    ) -> List[Dict[str, Any]]:
        where_clause = SemanticMetricsRAG._build_domain_layer_conditions(
            [], domain=domain, layer1=layer1, layer2=layer2
        )
        return self._search_metrics_details(query_text=query_text, where=where_clause, top_n=top_n)

    @classmethod
    def _build_domain_layer_conditions(
        cls, conditions: List[WhereExpr], domain: str = "", layer1: str = "", layer2: str = ""
    ) -> Optional[str]:
        if domain:
            conditions.append(eq("domain", domain))
        if layer1:
            conditions.append(eq("layer1", layer1))
        if layer2:
            conditions.append(eq("layer2", layer2))
        if not conditions:
            return None

        return build_where(And(conditions))

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

        metric_where_clause = SemanticMetricsRAG._build_domain_layer_conditions(conditions, domain, layer1, layer2)
        return self._search_metrics_details(query_text, metric_where_clause)

    def _search_metrics_details(
        self, query_text: str, where: Optional[WhereExpr] = None, top_n=5
    ) -> List[Dict[str, Any]]:
        logger.info(f"start to search metrics, metric_where: {where}, query_text: {query_text}")
        metric_search_results = self.metric_storage.search(
            query_txt=query_text,
            select_fields=["name", "llm_text"],
            top_n=top_n,
            where=where,
        )
        if metric_search_results is None or metric_search_results.num_rows == 0:
            logger.info("Metric search returned no results")
            return []
        try:
            return metric_search_results.select(["llm_text"]).to_pylist()
        except Exception as e:
            logger.warning(f"Failed to extract metric results, exception: {str(e)}")
            return []

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

    def get_semantic_model(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not select_fields:
            select_fields = [
                "semantic_model_name",
                "domain",
                "layer1",
                "layer2",
                "semantic_model_desc",
                "identifiers",
                "dimensions",
                "measures",
                "semantic_file_path",
                "catalog_name",
                "database_name",
                "schema_name",
                "table_name",
            ]
        results = self.semantic_model_storage._search_all(
            where=and_(
                eq("catalog_name", catalog_name or ""),
                eq("database_name", database_name or ""),
                eq("schema_name", schema_name or ""),
                eq("table_name", table_name or ""),
            ),
            select_fields=select_fields,
        )
        if results is None or results.num_rows == 0:
            return []
        return results.to_pylist()
