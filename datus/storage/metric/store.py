# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

import pyarrow as pa
from datus_storage_base.conditions import WhereExpr, in_

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import EmbeddingModel
from datus.storage.subject_tree.store import BaseSubjectEmbeddingStore, base_schema_columns
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MetricStorage(BaseSubjectEmbeddingStore):
    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        super().__init__(
            table_name="metrics",
            embedding_model=embedding_model,
            schema=pa.schema(
                base_schema_columns()  # Provides: name, subject_id, created_at
                + [
                    # -- Identity & Basic Info --
                    pa.field("id", pa.string()),  # Unique ID: "metric:dau"
                    pa.field("semantic_model_name", pa.string()),  # Source semantic model
                    # -- Retrieval Fields --
                    pa.field("description", pa.string()),  # For LLM reading (RAG) and vector search
                    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
                    # -- MetricFlow Specific Fields --
                    pa.field("metric_type", pa.string()),  # "simple" | "derived" | "ratio" | "cumulative"
                    pa.field("measure_expr", pa.string()),  # Underlying aggregation: "COUNT(DISTINCT user_id)"
                    pa.field("base_measures", pa.list_(pa.string())),  # Dependency measures: ["revenue", "orders"]
                    pa.field("dimensions", pa.list_(pa.string())),  # Available dimensions: ["platform", "country"]
                    pa.field("entities", pa.list_(pa.string())),  # Related entities: ["user", "order"]
                    # -- Database Context (for compatibility) --
                    pa.field("catalog_name", pa.string()),
                    pa.field("database_name", pa.string()),
                    pa.field("schema_name", pa.string()),
                    # -- Generated SQL --
                    pa.field("sql", pa.string()),  # SQL generated from query_metrics dry_run
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
        """Create scalar and FTS indices for better search performance."""
        self._ensure_table_ready()

        self._create_scalar_index("semantic_model_name")
        self._create_scalar_index("id")
        self._create_scalar_index("catalog_name")
        self._create_scalar_index("database_name")
        self._create_scalar_index("schema_name")

        self.create_subject_index()
        self.create_fts_index(["description", "name"])

    def batch_store_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Store multiple metrics in the database efficiently.

        Args:
            metrics: List of dictionaries containing metric data with required fields:
                - subject_path: List[str] - Subject hierarchy path for each metric (e.g., ['Finance', 'Revenue', 'Q1'])
                - semantic_model_name: str - Name of the semantic model
                - name: str - Name of the metric
                - description: str - Description for embedding and display
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

    def batch_upsert_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Upsert multiple metrics (update if id exists, insert if not).

        Args:
            metrics: List of dictionaries containing metric data with required fields:
                - subject_path: List[str] - Subject hierarchy path for each metric
                - id: str - Unique identifier for the metric (e.g., "metric:dau")
                - Other fields same as batch_store_metrics
        """
        if not metrics:
            return

        # Validate all metrics have required subject_path
        for metric in metrics:
            subject_path = metric.get("subject_path")
            if not subject_path:
                raise ValueError("subject_path is required in metric data")

        # Use base class batch_upsert method
        self.batch_upsert(metrics, on_column="id")

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

    def search_all(
        self,
        where: Optional[WhereExpr] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search all metrics with optional filtering and field selection.
        Returns a list of dictionaries (backward compatibility for autocomplete).
        """
        return self._search_all(where=where, select_fields=select_fields).to_pylist()

    def delete_metric(self, subject_path: List[str], name: str) -> Dict[str, Any]:
        """Delete metric by subject_path and name.

        This method:
        1. Queries the metric to get its yaml_path
        2. Deletes the metric from vector store
        3. Removes the metric entry from the yaml file (if yaml_path exists)

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the metric to delete

        Returns:
            Dict with 'success', 'message', and optional 'yaml_updated' fields

        Examples:
            result = storage.delete_metric(
                subject_path=['Finance', 'Revenue'],
                name='total_revenue'
            )
            # Returns: {'success': True, 'message': 'Deleted metric...', 'yaml_updated': True}
        """
        import os

        import yaml

        # First, query all matching metrics to get their yaml_paths before deleting
        full_path = subject_path.copy()
        full_path.append(name)
        metrics = self.search_all_metrics(subject_path=full_path, select_fields=["name", "yaml_path"])

        # Collect all unique yaml_paths from matching metrics
        yaml_paths = list({m.get("yaml_path") for m in metrics if m.get("yaml_path")})

        # Delete from vector store using base class method
        deleted = self.delete_entry(subject_path, name)

        if not deleted:
            return {
                "success": False,
                "message": f"Metric '{name}' not found under subject_path={'/'.join(subject_path)}",
            }

        result = {
            "success": True,
            "message": f"Deleted metric '{name}' from vector store",
            "yaml_updated": False,
        }

        # Handle yaml files for all matching metrics
        for yaml_path in yaml_paths:
            if not os.path.exists(yaml_path):
                continue
            try:
                # Read yaml file (supports multi-document format)
                with open(yaml_path, "r", encoding="utf-8") as f:
                    docs = list(yaml.safe_load_all(f))

                # Filter out the metric doc with matching name
                filtered_docs = []
                metric_removed = False
                for doc in docs:
                    if doc is None:
                        continue
                    # Check if this is a metric doc with the target name
                    if "metric" in doc and doc["metric"].get("name") == name:
                        logger.info(f"Removing metric '{name}' from yaml file: {yaml_path}")
                        metric_removed = True
                        continue
                    filtered_docs.append(doc)

                # Write back if we removed something
                if metric_removed:
                    if filtered_docs:
                        # Write remaining docs back to file
                        with open(yaml_path, "w", encoding="utf-8") as f:
                            yaml.safe_dump_all(filtered_docs, f, allow_unicode=True, sort_keys=False)
                        result["yaml_updated"] = True
                        result["message"] = f"Deleted metric '{name}' from vector store and yaml file(s)"
                        logger.info(f"Updated yaml file: {yaml_path}")
                    else:
                        # File is empty after removing the metric, delete the file
                        os.remove(yaml_path)
                        result["yaml_updated"] = True
                        result["yaml_deleted"] = True
                        result["message"] = f"Deleted metric '{name}' from vector store and removed empty yaml file"
                        logger.info(f"Deleted empty yaml file: {yaml_path}")

            except Exception as e:
                logger.error(f"Failed to update yaml file {yaml_path}: {e}")
                result["message"] = f"Deleted metric '{name}' from vector store, but failed to update yaml: {e}"
                result["yaml_error"] = str(e)

        return result


class MetricRAG:
    """RAG interface for metric operations."""

    def __init__(self, agent_config: AgentConfig, sub_agent_name: Optional[str] = None):
        from datus.storage.registry import get_rag_storage

        self.storage: MetricStorage = get_rag_storage(MetricStorage, "metric", agent_config, sub_agent_name, "metrics")

    def truncate(self) -> None:
        """Drop the metrics table and reset state."""
        self.storage.truncate()

    def store_batch(self, metrics: List[Dict[str, Any]]):
        logger.info(f"store metrics: {metrics}")
        self.storage.batch_store_metrics(metrics)

    def upsert_batch(self, metrics: List[Dict[str, Any]]):
        """Upsert metrics (update if id exists, insert if not)."""
        logger.info(f"upsert metrics: {metrics}")
        self.storage.batch_upsert_metrics(metrics)

    def search_all_metrics(
        self,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.storage.search_all_metrics(subject_path=subject_path, select_fields=select_fields)

    def after_init(self):
        self.storage.create_indices()

    def get_metrics_size(self):
        return self.storage.table_size()

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
        return self.storage.search_metrics(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
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
        return self.storage.search_all_metrics(subject_path=full_path)

    def create_indices(self):
        """Create indices for metric storage."""
        self.storage.create_indices()

    def delete_metric(self, subject_path: List[str], name: str) -> Dict[str, Any]:
        """Delete metric by subject_path and name.

        This method deletes the metric from vector store and removes the metric entry
        from the yaml file (if yaml_path exists).

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the metric to delete

        Returns:
            Dict with 'success', 'message', and optional 'yaml_updated' fields
        """
        return self.storage.delete_metric(subject_path, name)
