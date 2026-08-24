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
from typing import Any, Dict, List, Optional, Union

from datus.configuration.agent_config import AgentConfig
from datus.storage.datasource_scope import resolve_datasource_id
from datus.storage.metric.store import MetricStorage, build_metric_id
from datus.storage.semantic_dataset.store import (
    KIND_DATASET,
    KIND_FIELD,
    SemanticDatasetRAG,
    dataset_row_id,
    field_row_id,
)
from datus.storage.subject_tree.store import SubjectTreeStore
from datus.tools.semantic_tools.models import SemanticModelInfo
from datus.utils.exceptions import DatusException, ErrorCode
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
        self.datasource_id = resolve_datasource_id(agent_config)
        self.semantic_model_store: Optional[SemanticDatasetRAG] = None
        self.metric_store: Optional[MetricStorage] = None
        self.subject_tree_store: Optional[SubjectTreeStore] = None

    def _ensure_semantic_model_store(self) -> SemanticDatasetRAG:
        """Lazy init semantic model storage."""
        if self.semantic_model_store is None:
            self.semantic_model_store = SemanticDatasetRAG(self.agent_config, datasource_id=self.datasource_id)
        return self.semantic_model_store

    def _ensure_metric_store(self) -> MetricStorage:
        """Lazy init metric storage."""
        if self.metric_store is None:
            from datus.storage.metric.store import MetricRAG

            rag = MetricRAG(self.agent_config, datasource_id=self.datasource_id)
            self.metric_store = rag.storage
        return self.metric_store

    def _ensure_subject_tree_store(self) -> SubjectTreeStore:
        """Lazy init subject tree storage via registry singleton."""
        if self.subject_tree_store is None:
            from datus.storage.registry import get_subject_tree_store

            self.subject_tree_store = get_subject_tree_store(
                project=self.agent_config.project_name,
                datasource_id=self.datasource_id,
            )
        return self.subject_tree_store

    def _split_qualified_table_name(self, table_ref: str) -> tuple[str, dict[str, str]]:
        """Return (table leaf, coordinates) for a possibly qualified name."""
        empty = {"catalog_name": "", "database_name": "", "schema_name": ""}
        if "." not in table_ref:
            return table_ref, empty
        from datus.utils.sql_utils import parse_table_name_parts

        try:
            parsed = parse_table_name_parts(table_ref, dialect=self.agent_config.db_type or "snowflake")
        except Exception as exc:
            # An unparseable name is still better stored whole than dropped.
            logger.warning(f"Could not parse qualified table name {table_ref!r}: {exc}")
            return table_ref, empty
        return parsed.get("table_name") or table_ref, {key: parsed.get(key) or "" for key in empty}

    def store_semantic_model(
        self,
        model_data: Union[SemanticModelInfo, Dict[str, Any]],
    ) -> None:
        """
        Store semantic model to unified storage.

        Args:
            model_data: Either a SemanticModelInfo object or a dict with structure:
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
        # Convert SemanticModelInfo to dict format for storage
        if isinstance(model_data, SemanticModelInfo):
            extra = model_data.extra or {}
            table_name = model_data.table_name or extra.get("table_name")
            if not table_name:
                raise DatusException(
                    ErrorCode.SEMANTIC_ADAPTER_SYNC_FAILED,
                    message_args={
                        "error_message": f"SemanticModelInfo '{model_data.name}' missing physical table_name"
                    },
                )

            converted = {
                "semantic_model_name": model_data.name,
                "description": model_data.description or "",
                "table_name": table_name,
                "catalog_name": model_data.catalog_name or extra.get("catalog_name", ""),
                "database_name": model_data.database_name or extra.get("database_name", ""),
                "schema_name": model_data.schema_name or extra.get("schema_name", ""),
                "dimensions": [
                    {"name": d.name, "description": d.description or "", "expr": ""} for d in model_data.dimensions
                ],
                "measures": [{"name": m, "description": "", "expr": ""} for m in model_data.measures],
            }
            if model_data.platform_type:
                converted["platform_type"] = model_data.platform_type
            if model_data.extra:
                converted["extra"] = model_data.extra
            model_data = converted

        # Validate required field
        if "semantic_model_name" not in model_data:
            raise ValueError("model_data must contain 'semantic_model_name' field")

        rag = self._ensure_semantic_model_store()
        semantic_model_name = model_data["semantic_model_name"]
        raw_table_name = model_data.get("table_name", "")
        # An adapter model always binds a physical table. Without one the row
        # would look like a query-backed dataset and no table lookup could ever
        # reach it, so it is refused here as it is on the typed path above.
        if not raw_table_name:
            raise DatusException(
                ErrorCode.SEMANTIC_ADAPTER_SYNC_FAILED,
                message_args={"error_message": f"semantic model '{semantic_model_name}' missing physical table_name"},
            )
        # An adapter may report a qualified name. Table lookups match
        # ``source_table`` against the leaf a connector reports, so the
        # qualifiers belong in the coordinate columns -- the same split the
        # Dosi authoring path performs.
        table_name, parsed_parts = self._split_qualified_table_name(raw_table_name)
        catalog = model_data.get("catalog_name", "") or parsed_parts["catalog_name"]
        database = model_data.get("database_name", "") or parsed_parts["database_name"]
        schema = model_data.get("schema_name", "") or parsed_parts["schema_name"]
        updated_at = datetime.now().replace(microsecond=0)
        # An adapter reports one model per physical table, so the dataset takes
        # the table's name; datasets authored in Dosi carry their own.
        dataset_name = table_name
        coordinates = {"catalog_name": catalog, "database_name": database, "schema_name": schema}

        rows = [
            {
                "id": dataset_row_id(semantic_model_name, dataset_name),
                "kind": KIND_DATASET,
                "semantic_model_name": semantic_model_name,
                "dataset_name": dataset_name,
                "name": dataset_name,
                "source_table": table_name,
                "source_query": "",
                "description": model_data.get("description", ""),
                "search_text": " ".join(
                    part
                    for part in (semantic_model_name, dataset_name, table_name, model_data.get("description", ""))
                    if part
                ),
                "yaml_path": "",
                "updated_at": updated_at,
                **coordinates,
            }
        ]

        # An adapter's measures are not fields of the dataset — metrics live in
        # their own store — so only dimensions and identifiers become rows.
        field_groups = (
            ("dimensions", {"is_dimension": True}),
            ("identifiers", {"is_primary_key": True}),
        )
        counts = {}
        for key, flags in field_groups:
            entries = model_data.get(key, []) or []
            counts[key] = 0
            for entry in entries:
                if not isinstance(entry, dict) or "name" not in entry:
                    logger.warning(f"Skipping {key[:-1]} without 'name' field in model '{semantic_model_name}'")
                    continue
                field_name = entry["name"]
                rows.append(
                    {
                        "id": field_row_id(semantic_model_name, dataset_name, field_name),
                        "kind": KIND_FIELD,
                        "semantic_model_name": semantic_model_name,
                        "dataset_name": dataset_name,
                        "name": field_name,
                        "source_table": table_name,
                        "expr": entry.get("expr") or field_name,
                        "description": entry.get("description", ""),
                        "search_text": " ".join(
                            part for part in (dataset_name, field_name, entry.get("description", "")) if part
                        ),
                        "yaml_path": "",
                        "updated_at": updated_at,
                        **flags,
                        **coordinates,
                    }
                )
                counts[key] += 1

        # Keyed on storage_key so re-syncing an adapter reconciles its rows
        # instead of appending a second copy of every one of them.
        rag.upsert_batch(rows)

        logger.info(
            f"Stored semantic model '{semantic_model_name}': "
            f"{counts['dimensions']} dimensions, {counts['identifiers']} identifiers"
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

        metric_obj = {
            "subject_path": subject_path,
            "id": build_metric_id(subject_path, metric_data["name"]),
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

            for model_entry in models:
                try:
                    if isinstance(model_entry, SemanticModelInfo):
                        entry_extra = model_entry.extra or {}
                        table_name = model_entry.table_name or entry_extra.get("table_name")
                        if table_name:
                            self.store_semantic_model(model_entry)
                            stats["semantic_models_synced"] += 1
                        else:
                            model_data = adapter.get_semantic_model(table_name=model_entry.name)
                            if model_data:
                                self.store_semantic_model(model_data)
                                stats["semantic_models_synced"] += 1
                            else:
                                logger.warning(
                                    f"Skipping semantic model '{model_entry.name}' from "
                                    f"{adapter.service_type}: missing physical table_name metadata"
                                )
                    else:
                        model_data = adapter.get_semantic_model(table_name=model_entry)
                        if model_data:
                            self.store_semantic_model(model_data)
                            stats["semantic_models_synced"] += 1
                except Exception as e:
                    model_id = model_entry.name if isinstance(model_entry, SemanticModelInfo) else model_entry
                    logger.error(f"Failed to sync semantic model '{model_id}' from {adapter.service_type}: {e}")
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
