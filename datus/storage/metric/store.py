# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
from typing import Any, Dict, List, Optional

import pyarrow as pa

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import BaseEmbeddingStore, EmbeddingModel
from datus.storage.lancedb_conditions import And, and_, build_where, eq, in_
from datus.storage.subject_tree.store import BaseSubjectEmbeddingStore, base_schema_columns

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


class MetricStorage(BaseSubjectEmbeddingStore):
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        super().__init__(
            db_path=db_path,
            table_name="metrics",
            embedding_model=embedding_model,
            schema=pa.schema(
                base_schema_columns()
                + [
                    pa.field("semantic_model_name", pa.string()),
                    pa.field("llm_text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                ]
            ),
            vector_source_name="llm_text",
        )
        self.reranker = None

    def create_indices(self):
        """Create scalar and FTS indices for better search performance."""
        # Ensure table is ready before creating indices
        self._ensure_table_ready()

        # Create metric-specific scalar indices
        self.table.create_scalar_index("semantic_model_name", replace=True)

        # Use base class method for subject index
        self.create_subject_index()

        # Create FTS index for metric-specific fields
        self.create_fts_index(["name"])

    def batch_store_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Store multiple metrics in the database efficiently.

        Args:
            metrics: List of dictionaries containing metric data with required fields:
                - subject_path: List[str] - Subject hierarchy path for each metric (e.g., ['Finance', 'Revenue', 'Q1'])
                - semantic_model_name: str - Name of the semantic model
                - name: str - Name of the metric
                - llm_text: str - Text description for embedding
                - created_at: str - Creation timestamp (optional, will auto-generate if not provided)
        """
        if not metrics:
            return

        # Validate all metrics have required subject_path
        for metric in metrics:
            subject_path = metric.get("subject_path")
            if not subject_path:
                raise ValueError("subject_path is required in metric data")

        # Use base class batch_store method
        self.batch_store(metrics)

    def _search_metrics_internal(
        self,
        query_text: Optional[str] = None,
        semantic_model_names: Optional[List[str]] = None,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search metrics with semantic model and subject filtering."""
        # Build additional conditions for semantic model filtering
        additional_conditions = []
        if semantic_model_names:
            additional_conditions.append(in_("semantic_model_name", semantic_model_names))

        # Use base class method with metric-specific field selection
        return self.search_with_subject_filter(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
            name_field="name",
            additional_conditions=additional_conditions if additional_conditions else None,
            selected_fields=select_fields,
        )

    def search_all_metrics(
        self,
        semantic_model_names: Optional[List[str]] = None,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search all metrics with optional semantic model and subject filtering."""
        return self._search_metrics_internal(
            semantic_model_names=semantic_model_names, subject_path=subject_path, select_fields=select_fields
        )

    def search_metrics(
        self,
        query_text: str = "",
        semantic_model_names: Optional[List[str]] = None,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search metrics by query text with optional semantic model and subject filtering."""
        return self._search_metrics_internal(
            query_text=query_text,
            semantic_model_names=semantic_model_names,
            subject_path=subject_path,
            top_n=top_n,
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
        self.metric_storage.batch_store_metrics(metrics)

    def search_all_semantic_models(
        self, database_name: str, select_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        return self.semantic_model_storage.search_all(database_name, select_fields=select_fields)

    def search_all_metrics(self, select_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.metric_storage.search_all_metrics(select_fields=select_fields)

    def after_init(self):
        self.semantic_model_storage.create_indices()
        self.metric_storage.create_indices()

    def get_semantic_model_size(self):
        return self.semantic_model_storage.table_size()

    def get_metrics_size(self):
        return self.metric_storage.table_size()

    def search_metrics(
        self, query_text: str, subject_path: Optional[List[str]] = None, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Search metrics by query text with optional subject path filtering.

        Args:
            query_text: Query text to search for
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue'])
            top_n: Number of results to return

        Returns:
            List of matching metrics
        """
        return self.metric_storage.search_metrics(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
        )

    def search_hybrid_metrics(
        self,
        query_text: str,
        subject_path: Optional[List[str]] = None,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n: int = 5,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search hybrid metrics (semantic + metrics) with optional subject path filtering.

        Args:
            query_text: Query text to search for
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue'])
            catalog_name: Optional catalog name filter
            database_name: Optional database name filter
            schema_name: Optional schema name filter
            top_n: Number of results to return
            use_rerank: Whether to use reranking (currently unused)

        Returns:
            List of matching metrics
        """
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

        return self.metric_storage.search_metrics(
            query_text=query_text, semantic_model_names=semantic_names, subject_path=subject_path
        )

    def get_metrics_detail(self, subject_path: List[str], name: str) -> List[Dict[str, Any]]:
        """Get metrics detail by subject path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            name: Metric name

        Returns:
            List containing the matching metric entry details
        """
        full_path = subject_path.copy()
        full_path.append(name)
        return self.metric_storage.search_all_metrics(subject_path=full_path)

    def get_semantic_model(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get semantic model with optional filtering.

        Args:
            catalog_name: Optional catalog name filter
            database_name: Optional database name filter
            schema_name: Optional schema name filter
            table_name: Optional table name filter
            select_fields: Optional list of fields to select

        Returns:
            List of matching semantic models
        """
        if not select_fields:
            select_fields = [
                "semantic_model_name",
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
