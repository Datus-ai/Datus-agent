# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import os
import re
from typing import Any, Dict, List, Optional, Set

import pyarrow as pa
import yaml
from datus_storage_base.conditions import WhereExpr, and_, eq, in_, not_

from datus.configuration.agent_config import AgentConfig
from datus.storage.base import EmbeddingModel
from datus.storage.datasource_scope import datasource_condition, resolve_datasource_id
from datus.storage.fts import FtsField, FtsSpec
from datus.storage.knowledge_provenance import enrich_metric_results, is_knowledge_provenance_enabled
from datus.storage.subject_tree.store import BaseSubjectEmbeddingStore, base_schema_columns
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

METRIC_ID_PREFIX = "metric:"
_METRIC_DEFINITION_FIELDS = ("semantic_model_name", "metric_type", "measure_expr", "base_measures")
_METRIC_LOOKUP_MAX_QUERY_LENGTH = 120
_METRIC_EXACT_QUERY_PATTERN = re.compile(r"^[0-9A-Za-z_.:/-]+$")
_METRIC_LOOKUP_TOKEN_PATTERN = re.compile(r"[^0-9a-zA-Z]+")


def normalize_metric_name(value: Any) -> str:
    """Return the datasource-local business key used for metric names."""

    return str(value or "").strip().lower()


def metric_lookup_key(value: Any) -> str:
    """Normalize metric names and ids for exact-name retrieval fallback."""

    text = str(value or "").strip().lower()
    if text.startswith(METRIC_ID_PREFIX):
        text = text[len(METRIC_ID_PREFIX) :]
    text = _METRIC_LOOKUP_TOKEN_PATTERN.sub("_", text).strip("_")
    return re.sub(r"_+", "_", text)


def metric_exact_query_keys(query_text: Any) -> Set[str]:
    """Return lookup keys for identifier-like exact metric lookups."""

    text = str(query_text or "").strip()
    if not text or len(text) > _METRIC_LOOKUP_MAX_QUERY_LENGTH:
        return set()
    if not _METRIC_EXACT_QUERY_PATTERN.fullmatch(text):
        return set()

    key = metric_lookup_key(text)
    return {key} if key else set()


def metric_matches_exact_query(metric: Dict[str, Any], query_keys: Set[str]) -> bool:
    """Return true when a metric row exactly matches normalized query keys."""

    if not query_keys:
        return False
    for field in ("name", "id"):
        if metric_lookup_key(metric.get(field)) in query_keys:
            return True
    return False


def build_metric_id(subject_path: List[str], name: str) -> str:
    """Build the stable datasource-local business key for a metric.

    ``subject_path`` is kept in the signature for existing call sites, but a
    metric's identity must not change when it is moved in the subject tree.
    Datasource isolation is handled by row-level storage scope.
    """

    metric_name = str(name or "").strip()
    if not metric_name:
        raise ValueError("metric name is required")
    return f"{METRIC_ID_PREFIX}{metric_name}"


def _normalize_definition_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(item).strip() for item in value if str(item).strip()))
    return str(value).strip()


def metric_definition_conflict(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Optional[str]:
    """Return the first conflicting core definition field, if any.

    Empty values are treated as unknowns so legacy rows can be completed without
    blocking a safe upsert.
    """

    for field in _METRIC_DEFINITION_FIELDS:
        existing_value = _normalize_definition_value(existing.get(field))
        incoming_value = _normalize_definition_value(incoming.get(field))
        if existing_value and incoming_value and existing_value != incoming_value:
            return field
    return None


class MetricStorage(BaseSubjectEmbeddingStore):
    # Only fields editable through the CLI panel (MetricsPanel) are synced to YAML.
    # Other metric fields (metric_type, type_params, etc.) are not exposed for UI edits.
    _METRIC_DB_TO_YAML = {
        "description": "description",
    }

    def __init__(self, embedding_model: EmbeddingModel, **kwargs):
        super().__init__(
            table_name="metrics",
            embedding_model=embedding_model,
            schema=pa.schema(
                base_schema_columns()  # Provides: name, subject_id, created_at
                + [
                    # -- Identity & Basic Info --
                    pa.field("id", pa.string()),  # Datasource-local ID: "metric:dau"
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
                    # Semantic Hub stable node id (locked_metadata.uid); empty for
                    # metrics not governed by the Hub. Lets retrieval be counted
                    # per Hub uid (consumption tracking).
                    pa.field("uid", pa.string()),
                ]
            ),
            vector_source_name="description",
            vector_column_name="vector",
            unique_columns=["storage_key"],
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
        self.create_fts_index(FtsSpec((FtsField("name", boost=3.0), FtsField("description"))))

    def _scope_column_migration_exprs(self) -> Dict[str, str]:
        # Also backfill the Hub uid column on pre-existing metric tables (empty
        # default; real uids land on the next build-kb / pull vectorization),
        # reusing the same best-effort ensure_columns migration as scope columns.
        exprs = super()._scope_column_migration_exprs()
        if self._schema is not None and "uid" in set(self._schema.names):
            exprs = {**exprs, "uid": "''"}
        return exprs

    def batch_store_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Store multiple metrics in the database efficiently.

        Existing rows whose id embedded a subject path are reconciled by metric
        name before upsert: equivalent definitions are removed in favor of the
        datasource-local id, and conflicting definitions raise instead of
        creating duplicate business keys.

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

        prepared, stale_duplicate_ids = self._prepare_metrics_for_write(metrics, check_existing=True)
        if prepared:
            self.batch_upsert(prepared, on_column="storage_key")
        if stale_duplicate_ids:
            self._delete_rows(
                and_(
                    datasource_condition(self.datasource_id),
                    in_("id", stale_duplicate_ids),
                )
            )

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

        prepared, stale_duplicate_ids = self._prepare_metrics_for_write(metrics, check_existing=True)
        if prepared:
            self.batch_upsert(prepared, on_column="storage_key")
        if stale_duplicate_ids:
            self._delete_rows(
                and_(
                    datasource_condition(self.datasource_id),
                    in_("id", stale_duplicate_ids),
                )
            )

    def _prepare_metrics_for_write(
        self,
        metrics: List[Dict[str, Any]],
        check_existing: bool = False,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        prepared: List[Dict[str, Any]] = []
        index_by_name: Dict[str, int] = {}

        for metric in metrics:
            subject_path = metric.get("subject_path")
            if not subject_path:
                raise ValueError("subject_path is required in metric data")

            name = str(metric.get("name") or "").strip()
            if not name:
                raise ValueError("metric name is required in metric data")

            normalized_name = normalize_metric_name(name)
            updated = dict(metric)
            updated["name"] = name
            updated["id"] = build_metric_id(subject_path, name)

            existing_index = index_by_name.get(normalized_name)
            if existing_index is not None:
                existing_metric = prepared[existing_index]
                conflict_field = metric_definition_conflict(existing_metric, updated)
                if conflict_field:
                    raise ValueError(
                        f"Metric name conflict within datasource for '{name}': "
                        f"field '{conflict_field}' differs in the same write batch. "
                        "Metric names must be unique within a datasource."
                    )
                prepared[existing_index] = updated
                continue

            index_by_name[normalized_name] = len(prepared)
            prepared.append(updated)

        stale_duplicate_ids: List[str] = []
        if check_existing and prepared:
            stale_duplicate_ids = self._find_stale_existing_metric_ids(prepared)

        return prepared, stale_duplicate_ids

    def _find_stale_existing_metric_ids(self, metrics: List[Dict[str, Any]]) -> List[str]:
        names = {normalize_metric_name(metric.get("name")) for metric in metrics}
        fields = ["id", "name", *_METRIC_DEFINITION_FIELDS]
        existing_rows = self._search_all(
            where=datasource_condition(self.datasource_id),
            select_fields=fields,
        ).to_pylist()

        stale_duplicate_ids: List[str] = []
        for existing in existing_rows:
            existing_name = normalize_metric_name(existing.get("name"))
            if existing_name not in names:
                continue
            incoming = next(metric for metric in metrics if normalize_metric_name(metric.get("name")) == existing_name)
            conflict_field = metric_definition_conflict(existing, incoming)
            if conflict_field:
                raise ValueError(
                    f"Metric name conflict within datasource for '{incoming['name']}': "
                    f"existing metric id '{existing.get('id')}' has a different '{conflict_field}'. "
                    "Choose a more specific metric name or update the existing metric explicitly."
                )
            if existing.get("id") and existing.get("id") != incoming.get("id"):
                stale_duplicate_ids.append(str(existing["id"]))

        return stale_duplicate_ids

    def _search_metrics_internal(
        self,
        query_text: Optional[str] = None,
        semantic_model_names: Optional[List[str]] = None,
        subject_path: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        extra_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Search metrics with semantic model and subject filtering."""
        # Build additional conditions for semantic model filtering
        additional_conditions = []
        if semantic_model_names:
            additional_conditions.append(in_("semantic_model_name", semantic_model_names))
        if extra_conditions:
            additional_conditions.extend(extra_conditions)

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
        extra_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Search all metrics with optional semantic model and subject filtering."""
        return self._search_metrics_internal(
            semantic_model_names=semantic_model_names,
            subject_path=subject_path,
            select_fields=select_fields,
            extra_conditions=extra_conditions,
        )

    def search_metrics(
        self,
        query_text: str = "",
        semantic_model_names: Optional[List[str]] = None,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
        extra_conditions: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Search metrics by query text with optional semantic model and subject filtering."""
        return self._search_metrics_internal(
            query_text=query_text,
            semantic_model_names=semantic_model_names,
            subject_path=subject_path,
            top_n=top_n,
            extra_conditions=extra_conditions,
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

    def delete_metric(
        self,
        subject_path: List[str],
        name: str,
        extra_conditions: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Delete metric by subject_path and name.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the metric to delete
            extra_conditions: Additional filter conditions (e.g., datasource_id filter)
            datasource_id: Datasource identifier for tenant isolation

        Returns:
            Dict with 'success', 'message', and optional 'yaml_updated' fields
        """
        import os

        import yaml

        # First, query all matching metrics to get their yaml_paths before deleting
        full_path = subject_path.copy()
        full_path.append(name)
        metrics = self.search_all_metrics(
            subject_path=full_path,
            select_fields=["name", "yaml_path"],
            extra_conditions=extra_conditions,
        )

        # Collect all unique yaml_paths from matching metrics
        yaml_paths = list({m.get("yaml_path") for m in metrics if m.get("yaml_path")})

        # Delete from vector store using base class method
        deleted = self.delete_entry(subject_path, name, extra_conditions=extra_conditions)

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

    def update_entry(
        self,
        subject_path: List[str],
        name: str,
        update_values: Dict[str, Any],
        extra_conditions: Optional[List] = None,
    ) -> bool:
        """Update a metric in the vector DB and sync changes back to the YAML file.

        Args:
            subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue'])
            name: Name of the metric to update
            update_values: Dict of field names to new values
            extra_conditions: Additional filter conditions

        Returns:
            True if the update succeeded, False otherwise
        """
        # Query yaml_path BEFORE the update (same pattern as delete_metric)
        full_path = subject_path.copy()
        full_path.append(name)
        metrics = self.search_all_metrics(
            subject_path=full_path,
            select_fields=["name", "yaml_path"],
            extra_conditions=extra_conditions,
        )
        yaml_paths = list({m.get("yaml_path") for m in metrics if m.get("yaml_path")})

        # Update in vector store using base class method
        result = super().update_entry(subject_path, name, update_values, extra_conditions)
        if not result:
            return False

        # Sync changes to each yaml file
        for yaml_path in yaml_paths:
            self._sync_metric_update_to_yaml(yaml_path, name, update_values)

        return result

    def _sync_metric_update_to_yaml(self, yaml_path: str, name: str, update_values: Dict[str, Any]) -> None:
        """Sync metric field updates back to the YAML file.

        Args:
            yaml_path: Path to the YAML file containing the metric
            name: Name of the metric to update
            update_values: Dict of vector DB field names to new values
        """
        if not os.path.exists(yaml_path):
            return

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))

            # Filter out None docs (empty sections in multi-doc YAML)
            docs = [doc for doc in docs if doc is not None]

            updated = False
            for doc in docs:
                if doc.get("metric", {}).get("name") == name:
                    for db_key, yaml_key in self._METRIC_DB_TO_YAML.items():
                        if db_key in update_values:
                            doc["metric"][yaml_key] = update_values[db_key]
                    updated = True

            if updated:
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)
                logger.info(f"Updated metric '{name}' in yaml file: {yaml_path}")

        except Exception as e:
            logger.error(f"Failed to update yaml file {yaml_path}: {e}")

    def rename(self, old_path: List[str], new_path: List[str]) -> bool:
        """Rename or move a metric entry and sync subject_tree to the YAML file.

        When the parent subject path changes, update the metric's
        ``locked_metadata.tags`` entry to reflect the new subject_tree.

        Args:
            old_path: Current full path (subject_path + name)
            new_path: Target full path (subject_path + name)

        Returns:
            True on successful rename.
        """
        # Pre-query yaml_paths BEFORE the rename, using the old path
        yaml_paths: List[str] = []
        if len(old_path) >= 2:
            try:
                metrics = self.search_all_metrics(
                    subject_path=old_path,
                    select_fields=["name", "yaml_path"],
                )
                yaml_paths = list({m.get("yaml_path") for m in metrics if m.get("yaml_path")})
            except Exception as e:
                logger.warning(f"Failed to query yaml_path before metric rename: {e}")

        result = super().rename(old_path, new_path)

        # Sync subject_tree to YAML only when the parent path actually changes
        old_parent = old_path[:-1] if len(old_path) >= 2 else []
        new_parent = new_path[:-1] if len(new_path) >= 2 else []
        if result and old_parent != new_parent and yaml_paths:
            new_name = new_path[-1]
            for yaml_path in yaml_paths:
                self._sync_metric_subject_tree_to_yaml(yaml_path, new_name, new_parent)

        return result

    def _sync_metric_subject_tree_to_yaml(self, yaml_path: str, metric_name: str, new_parent_path: List[str]) -> None:
        """Update the metric's locked_metadata.tags subject_tree entry in YAML.

        Tag format: ``"subject_tree: path/component1/component2"``.
        Replaces an existing subject_tree tag if present, otherwise appends it.

        Args:
            yaml_path: Path to the YAML file containing the metric
            metric_name: Name of the metric to locate in the YAML
            new_parent_path: New subject path components (excluding the metric name)
        """
        if not os.path.exists(yaml_path):
            return

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))

            docs = [doc for doc in docs if doc is not None]
            new_tag = f"subject_tree: {'/'.join(new_parent_path)}"

            updated = False
            for doc in docs:
                if doc.get("metric", {}).get("name") != metric_name:
                    continue
                metric = doc["metric"]
                locked_metadata = metric.setdefault("locked_metadata", {})
                tags = locked_metadata.setdefault("tags", [])

                replaced = False
                for i, tag in enumerate(tags):
                    if isinstance(tag, str) and tag.startswith("subject_tree:"):
                        tags[i] = new_tag
                        replaced = True
                        break
                if not replaced:
                    tags.append(new_tag)
                updated = True

            if updated:
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump_all(docs, f, allow_unicode=True, sort_keys=False)
                logger.info(f"Updated subject_tree for metric '{metric_name}' in yaml file: {yaml_path}")

        except Exception as e:
            logger.error(f"Failed to sync subject_tree to yaml {yaml_path}: {e}")

    def sync_yaml_subject_tree_for_subtree(self, root_node_id: int) -> None:
        """Sync the ``subject_tree`` tag in metric YAML files for a whole subtree.

        Intended to be called AFTER a subject_tree node has been renamed or moved
        (via ``SubjectTreeStore.rename``). Walks ``root_node_id`` and all descendant
        nodes, re-computes each node's full path from the (already updated)
        subject_tree, and rewrites the ``locked_metadata.tags`` subject_tree entry
        for every metric whose ``subject_node_id`` matches.

        Vector DB rows are not touched here -- only the YAML files on disk.

        Args:
            root_node_id: ID of the renamed/moved subject node.
        """
        try:
            descendants = self.subject_tree.get_descendants(root_node_id)
        except Exception as e:
            logger.warning(f"Failed to enumerate descendants of node {root_node_id}: {e}")
            return

        node_ids = [root_node_id] + [d["node_id"] for d in descendants]

        for node_id in node_ids:
            try:
                new_parent_path = self.subject_tree.get_full_path(node_id)
            except Exception as e:
                logger.warning(f"Failed to compute full path for node {node_id}: {e}")
                continue

            if not new_parent_path:
                continue

            try:
                entries = self.list_entries(node_id)
            except Exception as e:
                logger.warning(f"Failed to list metrics under node {node_id}: {e}")
                continue

            for entry in entries:
                yaml_path = entry.get("yaml_path")
                name = entry.get("name")
                if yaml_path and name:
                    self._sync_metric_subject_tree_to_yaml(yaml_path, name, new_parent_path)


class MetricRAG:
    """RAG interface for metric operations.

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

        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.datasource_id = resolve_datasource_id(agent_config, datasource_id)
        self._provenance_enabled = is_knowledge_provenance_enabled(agent_config)
        self.storage: MetricStorage = get_storage(
            MetricStorage,
            "metric",
            project=agent_config.project_name,
            datasource_id=self.datasource_id,
        )
        self._sub_agent_filter = _build_sub_agent_filter(agent_config, sub_agent_name, self.storage, "metrics")

    def _sub_agent_conditions(self) -> List:
        """Build datasource and sub-agent filter conditions."""
        conditions = [datasource_condition(self.datasource_id)]
        if self._sub_agent_filter:
            conditions.append(self._sub_agent_filter)
        return conditions

    def _selected_fields_with_provenance_id(
        self, selected_fields: Optional[List[str]]
    ) -> tuple[Optional[List[str]], bool]:
        if not getattr(self, "_provenance_enabled", False) or selected_fields is None or "id" in selected_fields:
            return selected_fields, False
        return [*selected_fields, "id"], True

    def _enrich_metric_results(
        self, results: List[Dict[str, Any]], strip_internal_id: bool = False
    ) -> List[Dict[str, Any]]:
        if not getattr(self, "_provenance_enabled", False):
            return results

        enriched = enrich_metric_results(self.agent_config, results)
        if not strip_internal_id:
            return enriched

        cleaned: List[Dict[str, Any]] = []
        for item in enriched:
            if isinstance(item, dict):
                updated = dict(item)
                updated.pop("id", None)
                cleaned.append(updated)
            else:
                cleaned.append(item)
        return cleaned

    def truncate(self) -> None:
        """Delete all metrics for this datasource."""
        self.storage.delete_datasource_rows(self.datasource_id)

    def delete_artifact_rows(self, yaml_path: str) -> None:
        """Delete metric rows projected from a single YAML artifact."""
        if not yaml_path:
            return
        self.storage._delete_rows(and_(eq("yaml_path", yaml_path), *self._sub_agent_conditions()))

    def delete_artifact_rows_except(self, yaml_path: str, keep_ids: List[str]) -> None:
        """Delete stale metric rows for one YAML artifact after replacement succeeds."""
        if not yaml_path:
            return
        normalized_keep_ids = [row_id for row_id in keep_ids if row_id]
        if not normalized_keep_ids:
            self.delete_artifact_rows(yaml_path)
            return
        self.storage._delete_rows(
            and_(eq("yaml_path", yaml_path), not_(in_("id", normalized_keep_ids)), *self._sub_agent_conditions())
        )

    def list_artifact_rows(self, yaml_path: str) -> List[Dict[str, Any]]:
        """Return metric rows projected from a single YAML artifact."""
        if not yaml_path:
            return []
        return self.storage._search_all(
            where=and_(eq("yaml_path", yaml_path), *self._sub_agent_conditions())
        ).to_pylist()

    def restore_artifact_rows(self, yaml_path: str, rows: List[Dict[str, Any]]) -> None:
        """Restore one YAML artifact to a previously captured row snapshot."""
        if not yaml_path:
            return
        self.delete_artifact_rows(yaml_path)
        if rows:
            self.upsert_batch(rows)
        self.create_indices()

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
        fields, strip_internal_id = self._selected_fields_with_provenance_id(select_fields)
        results = self.storage.search_all_metrics(
            subject_path=subject_path,
            select_fields=fields,
            extra_conditions=self._sub_agent_conditions(),
        )
        return self._enrich_metric_results(results, strip_internal_id)

    def after_init(self):
        self.storage.create_indices()

    def get_metrics_size(self):
        from datus_storage_base.conditions import and_

        conditions = self._sub_agent_conditions()
        if not conditions:
            return self.storage._count_rows()
        where = conditions[0] if len(conditions) == 1 else and_(*conditions)
        return self.storage._count_rows(where=where)

    def search_metrics(
        self, query_text: str, subject_path: Optional[List[str]] = None, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Search metrics by query text with optional subject path filtering."""
        exact_results = self._search_exact_metric_matches(query_text, subject_path)
        vector_results = self.storage.search_metrics(
            query_text=query_text,
            subject_path=subject_path,
            top_n=top_n,
            extra_conditions=self._sub_agent_conditions(),
        )
        results = self._merge_search_results(exact_results, vector_results, top_n)
        enriched = self._enrich_metric_results(results)
        self._emit_retrieval(query_text, enriched)
        return enriched

    def _emit_retrieval(self, query_text: str, results: List[Dict[str, Any]]) -> None:
        # Fire-and-forget: report retrieved Hub metrics so the host can count
        # consumption. Never let hook errors break a search.
        try:
            from datus.api.hooks import MetricRetrievalEvent, get_metric_retrieval_hook

            hook = get_metric_retrieval_hook()
            if hook is None:
                return
            hook.on_retrieval(
                MetricRetrievalEvent(
                    query_text=query_text,
                    datasource_id=self.datasource_id,
                    project_name=getattr(self.agent_config, "project_name", None),
                    sub_agent_name=self.sub_agent_name,
                    # uid is the contract's string field — normalize missing/null
                    # to "" (empty = not Hub-governed) rather than forwarding None.
                    metrics=[{"id": r.get("id"), "name": r.get("name"), "uid": r.get("uid") or ""} for r in results],
                )
            )
        except Exception as exc:  # noqa: BLE001 — retrieval must never fail on telemetry
            logger.debug(f"metric retrieval hook failed: {exc}")

    def _search_exact_metric_matches(
        self,
        query_text: str,
        subject_path: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        query_keys = metric_exact_query_keys(query_text)
        if not query_keys:
            return []

        try:
            candidates = self.storage.search_all_metrics(
                subject_path=subject_path,
                extra_conditions=self._sub_agent_conditions(),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Unable to run exact metric-name fallback: %s", exc)
            return []

        return [metric for metric in candidates if metric_matches_exact_query(metric, query_keys)]

    @classmethod
    def _merge_search_results(
        cls,
        exact_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for metric in [*exact_results, *vector_results]:
            key = cls._search_result_identity(metric)
            if key in seen:
                continue
            seen.add(key)
            merged.append(metric)
            if top_n and len(merged) >= top_n:
                break
        return merged

    @staticmethod
    def _search_result_identity(metric: Dict[str, Any]) -> str:
        metric_id = metric.get("id")
        if metric_id:
            return f"id:{metric_id}"
        return f"row:{json.dumps(metric, sort_keys=True, default=str)}"

    def get_metrics_detail(self, subject_path: List[str], name: str) -> List[Dict[str, Any]]:
        """Get metrics detail by subject path and name."""
        full_path = subject_path.copy()
        full_path.append(name)
        results = self.storage.search_all_metrics(
            subject_path=full_path,
            extra_conditions=self._sub_agent_conditions(),
        )
        return self._enrich_metric_results(results)

    def create_indices(self):
        """Create indices for metric storage."""
        self.storage.create_indices()

    def delete_metric(self, subject_path: List[str], name: str) -> Dict[str, Any]:
        """Delete metric by subject_path and name."""
        return self.storage.delete_metric(
            subject_path,
            name,
            extra_conditions=self._sub_agent_conditions(),
        )
