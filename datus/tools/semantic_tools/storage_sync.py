# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Storage Manager

Responsibilities:
1. Sync semantic models from adapters to SemanticModelStorage
2. Sync metrics from adapters to MetricStorage
3. Provide conversion between adapter formats and storage schemas
4. Manage subject tree assignments for metrics
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricStorage
from datus.storage.semantic_model.store import SemanticModelStorage
from datus.storage.subject_tree.store import SubjectTreeStore
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SemanticStorageManager:
    """Manages sync between semantic adapters and unified storage."""

    def __init__(self, agent_config: AgentConfig):
        """
        Initialize storage manager.

        Args:
            agent_config: Agent configuration
        """
        self.agent_config = agent_config
        self.semantic_model_store: Optional[SemanticModelStorage] = None
        self.metric_store: Optional[MetricStorage] = None
        self.subject_tree_store: Optional[SubjectTreeStore] = None

    def _ensure_semantic_model_store(self) -> SemanticModelStorage:
        """Lazy init semantic model storage."""
        if self.semantic_model_store is None:
            from datus.storage.semantic_model.rag import SemanticModelRAG

            rag = SemanticModelRAG(self.agent_config)
            self.semantic_model_store = rag.store
        return self.semantic_model_store

    def _ensure_metric_store(self) -> MetricStorage:
        """Lazy init metric storage."""
        if self.metric_store is None:
            from datus.storage.metric.rag import MetricRAG

            rag = MetricRAG(self.agent_config)
            self.metric_store = rag.store
        return self.metric_store

    def _ensure_subject_tree_store(self) -> SubjectTreeStore:
        """Lazy init subject tree storage."""
        if self.subject_tree_store is None:
            self.subject_tree_store = SubjectTreeStore(db_path=self.agent_config.db_path)
        return self.subject_tree_store

    def store_semantic_model(
        self,
        model_data: Dict[str, Any],
    ) -> None:
        """
        Store semantic model to unified storage.

        Args:
            model_data: Semantic model data with structure:
                {
                    "semantic_model_name": str,
                    "description": str,
                    "table_name": str,  # Physical table name
                    "catalog_name": str (optional),
                    "database_name": str (optional),
                    "schema_name": str (optional),
                    "dimensions": List[{name, description, expr}],
                    "measures": List[{name, description, expr}],
                    "identifiers": List[{name, description, expr}] (optional),
                }
        """
        # Validate required field
        if "semantic_model_name" not in model_data:
            raise ValueError("model_data must contain 'semantic_model_name' field")

        store = self._ensure_semantic_model_store()
        semantic_model_name = model_data["semantic_model_name"]
        table_name = model_data.get("table_name", "")
        catalog = model_data.get("catalog_name", "")
        database = model_data.get("database_name", "")
        schema = model_data.get("schema_name", "")
        updated_at = datetime.now()

        # Build fully qualified table name (filter empty parts)
        table_fq_name = ".".join(p for p in [catalog, database, schema, table_name] if p)

        # Store table object
        table_id = f"table:{table_fq_name}"
        table_obj = {
            "id": table_id,
            "kind": "table",
            "name": table_name,
            "fq_name": table_fq_name,
            "semantic_model_name": semantic_model_name,
            "catalog_name": catalog,
            "database_name": database,
            "schema_name": schema,
            "table_name": table_name,
            "description": model_data.get("description", ""),
            "is_dimension": False,
            "is_measure": False,
            "is_entity_key": False,
            "is_deprecated": False,
            "yaml_path": "",
            "updated_at": updated_at,
        }
        store.batch_store([table_obj])

        # Store dimensions
        dimensions = model_data.get("dimensions", [])
        dim_objects = []
        for dim in dimensions:
            # Skip dimensions without name field
            if not isinstance(dim, dict) or "name" not in dim:
                logger.warning(f"Skipping dimension without 'name' field in model '{semantic_model_name}'")
                continue
            dim_fq_name = f"{table_fq_name}.{dim['name']}"
            dim_id = f"column:{dim_fq_name}"
            dim_obj = {
                "id": dim_id,
                "kind": "column",
                "name": dim["name"],
                "fq_name": dim_fq_name,
                "semantic_model_name": semantic_model_name,
                "catalog_name": catalog,
                "database_name": database,
                "schema_name": schema,
                "table_name": table_name,
                "description": dim.get("description", ""),
                "is_dimension": True,
                "is_measure": False,
                "is_entity_key": False,
                "is_deprecated": False,
                "yaml_path": "",
                "updated_at": updated_at,
            }
            dim_objects.append(dim_obj)
        if dim_objects:
            store.batch_store(dim_objects)

        # Store measures
        measures = model_data.get("measures", [])
        measure_objects = []
        for measure in measures:
            # Skip measures without name field
            if not isinstance(measure, dict) or "name" not in measure:
                logger.warning(f"Skipping measure without 'name' field in model '{semantic_model_name}'")
                continue
            measure_fq_name = f"{table_fq_name}.{measure['name']}"
            measure_id = f"column:{measure_fq_name}"
            measure_obj = {
                "id": measure_id,
                "kind": "column",
                "name": measure["name"],
                "fq_name": measure_fq_name,
                "semantic_model_name": semantic_model_name,
                "catalog_name": catalog,
                "database_name": database,
                "schema_name": schema,
                "table_name": table_name,
                "description": measure.get("description", ""),
                "is_dimension": False,
                "is_measure": True,
                "is_entity_key": False,
                "is_deprecated": False,
                "yaml_path": "",
                "updated_at": updated_at,
            }
            measure_objects.append(measure_obj)
        if measure_objects:
            store.batch_store(measure_objects)

        # Store identifiers (entity keys)
        identifiers = model_data.get("identifiers", [])
        identifier_objects = []
        for identifier in identifiers:
            # Skip identifiers without name field
            if not isinstance(identifier, dict) or "name" not in identifier:
                logger.warning(f"Skipping identifier without 'name' field in model '{semantic_model_name}'")
                continue
            identifier_fq_name = f"{table_fq_name}.{identifier['name']}"
            identifier_id = f"column:{identifier_fq_name}"
            identifier_obj = {
                "id": identifier_id,
                "kind": "column",
                "name": identifier["name"],
                "fq_name": identifier_fq_name,
                "semantic_model_name": semantic_model_name,
                "catalog_name": catalog,
                "database_name": database,
                "schema_name": schema,
                "table_name": table_name,
                "description": identifier.get("description", ""),
                "is_dimension": False,
                "is_measure": False,
                "is_entity_key": True,
                "is_deprecated": False,
                "yaml_path": "",
                "updated_at": updated_at,
            }
            identifier_objects.append(identifier_obj)
        if identifier_objects:
            store.batch_store(identifier_objects)

        logger.info(
            f"Stored semantic model '{semantic_model_name}': "
            f"{len(dimensions)} dimensions, {len(measures)} measures, {len(identifiers)} identifiers"
        )

    def store_metric(
        self,
        metric_data: Dict[str, Any],
        subject_path: Optional[List[str]] = None,
    ) -> None:
        """
        Store metric to unified storage.

        Args:
            metric_data: Metric data with structure:
                {
                    "name": str,
                    "description": str,
                    "metric_type": str (optional),
                    "dimensions": List[str] (optional),
                    "measures": List[str] (optional),
                    "entities": List[str] (optional),
                    "unit": str (optional),
                    "format": str (optional),
                    "semantic_model_name": str (optional),
                }
            subject_path: Subject tree path (e.g., ["Finance", "Revenue", "Q1"])
        """
        # Validate required field
        if "name" not in metric_data:
            raise ValueError("metric_data must contain 'name' field")

        store = self._ensure_metric_store()

        # Use provided subject_path or default category
        if not subject_path:
            subject_path = ["Uncategorized"]

        # Include subject_path in ID to avoid collision for metrics with same name
        subject_path_str = "/".join(subject_path)
        metric_obj = {
            "subject_path": subject_path,
            "id": f"metric:{subject_path_str}.{metric_data['name']}",
            "name": metric_data["name"],
            "description": metric_data.get("description", ""),
            "semantic_model_name": metric_data.get("semantic_model_name", ""),
            "metric_type": metric_data.get("metric_type", "simple"),
            "measure_expr": "",  # Will be populated by specific adapters
            "base_measures": metric_data.get("measures", []),
            "dimensions": metric_data.get("dimensions", []),
            "entities": metric_data.get("entities", []),
            "catalog_name": metric_data.get("catalog_name", ""),
            "database_name": metric_data.get("database_name", ""),
            "schema_name": metric_data.get("schema_name", ""),
            "updated_at": datetime.now(),
        }

        store.batch_store_metrics([metric_obj])
        logger.debug(f"Stored metric '{metric_data['name']}' with subject path {subject_path}")

    async def sync_from_adapter(
        self,
        adapter: "BaseSemanticAdapter",  # noqa: F821
        sync_semantic_models: bool = True,
        sync_metrics: bool = True,
        subject_path: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Sync data from adapter to unified storage.

        Args:
            adapter: Semantic adapter instance
            sync_semantic_models: Whether to sync semantic models
            sync_metrics: Whether to sync metrics
            subject_path: Subject tree path for metrics categorization

        Returns:
            Statistics: {
                "semantic_models_synced": int,
                "metrics_synced": int,
            }
        """
        stats = {"semantic_models_synced": 0, "metrics_synced": 0}

        # Sync semantic models
        if sync_semantic_models:
            try:
                models = adapter.list_semantic_models()
            except Exception as e:
                logger.error(f"Failed to list semantic models from {adapter.service_type}: {e}")
                models = []

            for model_name in models:
                try:
                    model_data = adapter.get_semantic_model(table_name=model_name)
                    if model_data:
                        self.store_semantic_model(model_data)
                        stats["semantic_models_synced"] += 1
                except Exception as e:
                    logger.error(f"Failed to sync semantic model '{model_name}' from {adapter.service_type}: {e}")
                    continue

        # Sync metrics
        if sync_metrics:
            try:
                metrics = await adapter.list_metrics()
            except Exception as e:
                logger.error(f"Failed to list metrics from {adapter.service_type}: {e}")
                metrics = []

            for metric in metrics:
                try:
                    # Use metric's own path if available, otherwise use provided subject_path
                    metric_subject_path = metric.path if metric.path else subject_path

                    self.store_metric(
                        {
                            "name": metric.name,
                            "description": metric.description,
                            "metric_type": metric.type or "simple",
                            "dimensions": metric.dimensions,
                            "measures": metric.measures,
                            "unit": metric.unit,
                            "format": metric.format,
                        },
                        subject_path=metric_subject_path,
                    )
                    stats["metrics_synced"] += 1
                except Exception as e:
                    metric_id = getattr(metric, "name", "unknown")
                    logger.error(f"Failed to sync metric '{metric_id}' from {adapter.service_type}: {e}")
                    continue

        logger.info(
            f"Synced from {adapter.service_type}: "
            f"{stats['semantic_models_synced']} semantic models, "
            f"{stats['metrics_synced']} metrics"
        )

        return stats
