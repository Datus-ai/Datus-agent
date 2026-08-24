# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import json
from collections.abc import Iterable
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml
from agents import Tool
from datus_storage_base.conditions import And, eq

from datus.configuration.agent_config import AgentConfig
from datus.storage.artifact_replacement import (
    delete_stale_artifact_rows,
    restore_artifact_replacements,
    snapshot_artifact_replacements,
)
from datus.storage.metric.store import MetricRAG, build_metric_id, metric_definition_conflict, normalize_metric_name
from datus.storage.semantic_dataset.store import (
    KIND_DATASET,
    KIND_FIELD,
    KIND_RELATIONSHIP,
    SemanticDatasetRAG,
    _identifier_variants,
    _normalized_identifier,
    dataset_row_id,
    field_row_id,
    relationship_row_id,
)
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState


def _rows_to_dicts(rows: Any) -> List[Dict[str, Any]]:
    """Normalize storage row containers to a list of dictionaries."""

    if rows is None:
        return []
    if hasattr(rows, "to_pylist"):
        rows = rows.to_pylist()
    if isinstance(rows, dict):
        return [rows]
    if isinstance(rows, list):
        iterable: Iterable[Any] = rows
    elif isinstance(rows, tuple):
        iterable = rows
    elif isinstance(rows, Iterable) and not isinstance(rows, (str, bytes)):
        iterable = rows
    else:
        return []
    return [row for row in iterable if isinstance(row, dict)]


def _is_supported_row_container(rows: Any) -> bool:
    if rows is None:
        return True
    if hasattr(rows, "to_pylist"):
        return True
    if isinstance(rows, (dict, list, tuple)):
        return True
    return isinstance(rows, Iterable) and not isinstance(rows, (str, bytes))


_DB_PART_FIELDS = ("catalog_name", "database_name", "schema_name")


def _rag_scope_conditions(rag: Any) -> List[Any]:
    method = getattr(rag, "_sub_agent_conditions", None)
    if not callable(method):
        return []
    try:
        conditions = method()
    except Exception:
        return []
    return conditions if isinstance(conditions, list) else []


class GenerationTools:
    """
    Tools for semantic model generation workflow.

    This class provides tools for checking existing semantic models and
    completing the generation process.
    """

    permission_category: str = "semantic_tools"

    def __init__(
        self,
        agent_config: AgentConfig,
        generation_evidence: Optional[GenerationEvidence] = None,
        authoring_format: Optional[str] = None,
        osi_target_state: Optional["OsiSemanticModelTargetState"] = None,
        require_bound_osi_target: bool = False,
    ):
        self.agent_config = agent_config
        self.generation_evidence = generation_evidence or GenerationEvidence()
        if authoring_format:
            self.authoring_format = str(authoring_format).strip().lower()
        else:
            from datus.agent.node.semantic_authoring import resolve_authoring_format

            self.authoring_format = resolve_authoring_format(agent_config)
        self.metric_rag = MetricRAG(agent_config)
        self.semantic_dataset_rag = None
        if isinstance(getattr(agent_config, "project_name", ""), str):
            try:
                self.semantic_dataset_rag = SemanticDatasetRAG(agent_config)
            except Exception as exc:
                logger.debug(f"Failed to initialize semantic dataset storage: {exc}")
        self._semantic_object_exists_cache: Dict[tuple[str, str, str], FuncToolResult] = {}
        self._semantic_table_object_index: Optional[Dict[str, Dict[str, object]]] = None
        self.osi_target_state = osi_target_state
        self.require_bound_osi_target = require_bound_osi_target

    def _is_osi_authoring(self) -> bool:
        return self.authoring_format == "osi"

    def available_tools(self) -> List[Tool]:
        """
        Provide tools for generation workflow.

        Returns:
            List of available tools for generation workflow
        """
        return [
            trans_to_function_tool(func)
            for func in (
                self.check_semantic_object_exists,
                self.generate_sql_summary_id,
                self.publish_semantic_model,
                self.publish_metrics,
            )
        ]

    def check_semantic_object_exists(
        self,
        name: str = "",
        kind: str = "table",  # table, column, metric
        table_context: str = "",
        object_name: str = "",
    ) -> FuncToolResult:
        """
        Check if a semantic object (table, column, metric) already exists in vector store.

        Use this tool to avoid duplicating work.

        Args:
            name: Name of the object (e.g. "orders", "orders.amount")
            kind: Type of object ("table", "column", "metric")
            table_context: If checking a column/metric, providing the table name helps narrow search.
            object_name: Backward-compatible alias for name.

        Returns:
            dict: Check results containing existence status and details.
        """
        try:
            object_name = str(name or object_name or "").strip()
            if not object_name:
                return FuncToolResult(success=0, error="name is required")

            normalized_kind = str(kind or "").strip().lower()
            if self._is_osi_authoring() and self.require_bound_osi_target:
                return self._check_bound_osi_object(object_name, normalized_kind)
            cache_key = (
                normalized_kind,
                object_name.lower(),
                str(table_context or "").strip().lower(),
            )
            cached = self._semantic_object_exists_cache.get(cache_key)
            if cached is not None:
                logger.debug("check_semantic_object_exists cache hit: %s", cache_key)
                return cached.model_copy(deep=True)

            # Extract the final segment as target name (e.g., "public.orders" -> "orders")
            target_name = object_name.split(".")[-1].strip('`"[]').lower()

            found_object = None

            if normalized_kind == "table":
                table_index = self._get_semantic_table_object_index()
                for candidate in _identifier_variants(object_name):
                    found_object = table_index.get(candidate) or table_index.get(_normalized_identifier(candidate))
                    if found_object:
                        break
            elif normalized_kind == "metric":
                # Exact match for metric using SQL WHERE condition
                storage = self.metric_rag.storage
                where = And([eq("name", target_name)] + _rag_scope_conditions(self.metric_rag))
                results = _rows_to_dicts(storage.search_all(where=where, select_fields=["id", "name"]))
                if results:
                    found_object = results[0]
            else:
                # An existence check is an exact lookup, not a similarity
                # question: scan the field rows in scope and compare names
                # case-insensitively rather than ranking by embedding.
                storage = self.semantic_dataset_rag.storage
                results = storage._search_all(
                    where=And([eq("kind", KIND_FIELD)] + _rag_scope_conditions(self.semantic_dataset_rag)),
                    select_fields=["id", "name", "kind", "dataset_name"],
                )
                # Determine the owning dataset from explicit context or dotted name
                target_dataset = None
                if table_context:
                    target_dataset = table_context.lower()
                elif "." in object_name:
                    target_dataset = object_name.rsplit(".", 1)[0].lower()

                for obj in _rows_to_dicts(results):
                    name_match = str(obj.get("name") or "").lower() == target_name
                    if not name_match:
                        continue
                    if target_dataset and str(obj.get("dataset_name") or "").lower() != target_dataset:
                        continue
                    found_object = obj
                    break

            if found_object:
                result = FuncToolResult(
                    result={
                        "exists": True,
                        "id": found_object.get("id"),
                        "name": found_object.get("name"),
                        "kind": found_object.get("kind") or normalized_kind,
                        "message": f"Object '{object_name}' ({normalized_kind}) already exists.",
                    }
                )
                self._semantic_object_exists_cache[cache_key] = result.model_copy(deep=True)
                return result

            result = FuncToolResult(
                result={"exists": False, "message": f"No {normalized_kind} found for '{object_name}'"}
            )
            self._semantic_object_exists_cache[cache_key] = result.model_copy(deep=True)
            return result

        except Exception as e:
            logger.error(f"Error checking semantic object existence: {e}")
            return FuncToolResult(success=0, error=f"Failed to check object: {str(e)}")

    def _check_bound_osi_object(self, object_name: str, normalized_kind: str) -> FuncToolResult:
        """Check the exact bound YAML instead of potentially stale vector storage."""
        state = self.osi_target_state
        if state is None or state.bound is None:
            return FuncToolResult(
                success=0,
                error="Bind an OSI semantic model before checking semantic objects.",
                result={"code": "semantic_model_required"},
            )
        path = str(state.bound["absolute_path"])
        try:
            state.require_current_revision(path)
            document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            models = document.get("semantic_model") if isinstance(document, dict) else None
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
                raise ValueError("The bound YAML must contain exactly one semantic model.")
            model = models[0]
            exists = False
            if normalized_kind == "metric":
                target_name = normalize_metric_name(object_name)
                exists = any(
                    isinstance(metric, dict) and normalize_metric_name(metric.get("name")) == target_name
                    for metric in model.get("metrics") or []
                )
            elif normalized_kind == "table":
                from datus.agent.node.semantic_authoring import _model_covers_table

                exists = _model_covers_table(state.bound, object_name)
            else:
                return FuncToolResult(
                    success=0,
                    error="Bound OSI YAML checks currently support kind='table' or kind='metric'.",
                )
            return FuncToolResult(
                result={
                    "exists": exists,
                    "name": object_name if exists else None,
                    "kind": normalized_kind,
                    "semantic_model_name": state.bound["semantic_model_name"],
                    "semantic_model_file": state.bound["semantic_model_file"],
                    "message": (
                        f"Object '{object_name}' ({normalized_kind}) exists in the bound OSI model."
                        if exists
                        else f"No {normalized_kind} found for '{object_name}' in the bound OSI model."
                    ),
                }
            )
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Failed to inspect the bound OSI semantic model: {exc}")

    # Backward compatibility wrapper
    def check_semantic_model_exists(
        self,
        table_name: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> FuncToolResult:
        """Legacy wrapper for checking table existence."""
        return self.check_semantic_object_exists(table_name, kind="table")

    def _get_semantic_table_object_index(self) -> Dict[str, Dict[str, object]]:
        """Load table semantic objects once and index all identifier variants."""

        if self._semantic_table_object_index is not None:
            return self._semantic_table_object_index

        storage = self.semantic_dataset_rag.storage
        select_fields = ["id", "name", "kind", "dataset_name", "source_table", *_DB_PART_FIELDS]
        rows = storage._search_all(
            where=And([eq("kind", KIND_DATASET)] + _rag_scope_conditions(self.semantic_dataset_rag)),
            select_fields=select_fields,
        )
        index: Dict[str, Dict[str, object]] = {}
        for obj in _rows_to_dicts(rows):
            # A dataset answers to its own name, to the table it binds, and to
            # that table's qualified name.
            source_table = str(obj.get("source_table") or "").strip()
            qualified = ".".join(
                part
                for part in (*(str(obj.get(field) or "").strip() for field in _DB_PART_FIELDS), source_table)
                if part
            )
            for value in (obj.get("dataset_name"), source_table, qualified):
                text = str(value or "").strip()
                if not text:
                    continue
                for variant in _identifier_variants(text):
                    if variant:
                        index.setdefault(variant, obj)
                    normalized = _normalized_identifier(variant)
                    if normalized:
                        index.setdefault(normalized, obj)

        self._semantic_table_object_index = index
        return index

    def resolve_planned_osi_semantic_target(self) -> tuple[str, str, str]:
        """Return the planned file, absolute path, and declared OSI model name."""
        planned = self.osi_target_state.planned if self.osi_target_state is not None else None
        if planned is None:
            raise ValueError(
                "Plan the OSI semantic-model name and file with plan_osi_semantic_model_target before publishing."
            )
        if self.osi_target_state.last_error_code:
            raise ValueError(
                "The OSI semantic-model plan is unresolved after a failed replan. "
                "Plan the authored target again before publishing."
            )

        semantic_model_file = str(planned.get("semantic_model_file") or "")
        resolved = self._resolve_generation_path(semantic_model_file, "semantic")
        if not resolved:
            raise ValueError(f"semantic_model_file escapes Knowledge Base sandbox: {semantic_model_file!r}")

        model_names = self.extract_osi_model_names(resolved)
        if len(model_names) != 1:
            raise ValueError(
                f"Generated OSI files must declare exactly one semantic model; found {model_names or '<none>'}."
            )

        planned_name = str(planned.get("semantic_model_name") or "")
        if model_names[0] != planned_name:
            raise ValueError(f"The planned OSI target is {planned_name!r}, but the file declares {model_names[0]!r}.")
        return semantic_model_file, resolved, model_names[0]

    def publish_semantic_model(self, semantic_model_files: List[str]) -> FuncToolResult:
        """Validate publication evidence and sync semantic artifacts to storage.

        Args:
            semantic_model_files: List of generated semantic model YAML file paths.
                Relative file names within the sub-agent's semantic-model workspace
                are preferred (e.g. ``["orders.yml", "customers.yml"]``).

        Returns:
            dict: Result containing completion message and semantic_model_files
        """
        try:
            if not self._is_osi_authoring():
                from datus.agent.node.semantic_authoring import QUERY_ONLY_MIGRATION_MESSAGE

                return FuncToolResult(success=0, error=QUERY_ONLY_MIGRATION_MESSAGE)

            semantic_model_file, resolved, model_name = self.resolve_planned_osi_semantic_target()
            semantic_model_files = [semantic_model_file]
            osi_target: tuple[str, str] = (resolved, model_name)
            validation_passed = self.generation_evidence.semantic_artifact_validation_passed(model_name, resolved)

            if not validation_passed:
                return FuncToolResult(
                    success=0,
                    error=(
                        "validate_semantic must pass for the exact semantic model artifact before publishing. "
                        "Call validate_semantic with the target semantic_model_name after the final file edit, "
                        "then retry publish_semantic_model."
                    ),
                    result={"semantic_model_files": semantic_model_files},
                )

            self._semantic_object_exists_cache.clear()
            self._semantic_table_object_index = None

            resolved, _model_name = osi_target
            sync_result = self.sync_osi_to_db(
                resolved,
                include_semantic_objects=True,
                include_metrics=False,
            )
            sync_results = [sync_result]
            if not sync_result.get("success"):
                return FuncToolResult(
                    success=0,
                    error=f"OSI semantic model KB sync failed: {sync_result.get('error', 'unknown')}",
                    result={"semantic_model_files": semantic_model_files, "sync": sync_results},
                )
            self.generation_evidence.mark_kb_sync("semantic")
            if self.osi_target_state is not None:
                self.osi_target_state.clear_artifact_snapshot()
            return FuncToolResult(
                result={
                    "message": f"Semantic model generation completed and synced {len(sync_results)} OSI file(s)",
                    "semantic_model_files": semantic_model_files,
                    "sync": sync_results,
                }
            )

        except Exception as e:
            logger.error(f"Error completing semantic model generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

    def publish_metrics(self, metric_file: str) -> FuncToolResult:
        """Validate publication evidence and sync metric artifacts to storage.

        Args:
            metric_file: Path to the generated metric YAML file (required).
                Relative paths (e.g. ``"metrics/orders_metrics.yml"``) are preferred
                and resolved against the sub-agent's semantic-model workspace using
                the live ``agent_config.current_datasource``. Absolute paths are only
                accepted when they resolve inside the Knowledge Base semantic-model
                sandbox.
        Returns:
            dict: Result containing completion message, file paths, metric SQLs, and sync status
        """
        try:
            if not self._is_osi_authoring():
                from datus.agent.node.semantic_authoring import QUERY_ONLY_MIGRATION_MESSAGE

                return FuncToolResult(success=0, error=QUERY_ONLY_MIGRATION_MESSAGE)

            metric_sqls = dict(self.generation_evidence.metric_sqls)
            # OSI authoring normally owns the metrics collection. When it
            # narrowly repairs a dataset for the requested metrics, publish
            # the same bound artifact as semantic input so KB profiles stay in
            # sync with the final YAML.
            semantic_model_files: List[str] = []

            exact_osi_target_required = self.require_bound_osi_target
            osi_metric_names_to_sync: Optional[set[str]] = None
            osi_touched_metric_names: Optional[set[str]] = None
            osi_absent_metric_names: set[str] = set()
            if exact_osi_target_required:
                state = self.osi_target_state
                if state is None or state.bound is None:
                    return FuncToolResult(
                        success=0,
                        error="Bind an existing OSI semantic model before publishing metrics.",
                        result={"code": "semantic_model_required"},
                    )
                if state.last_error_code:
                    return FuncToolResult(
                        success=0,
                        error=(
                            "The OSI target selection is unresolved after a failed bind. "
                            "Bind a valid target again before publishing."
                        ),
                        result={"code": state.last_error_code},
                    )
                abs_bound_metric = self._resolve_generation_path(metric_file, "metric")
                try:
                    state.require_current_revision(abs_bound_metric)
                except ValueError as exc:
                    return FuncToolResult(
                        success=0,
                        error=str(exc),
                        result={"code": "semantic_model_target_invalid"},
                    )
                if self.extract_osi_model_names(abs_bound_metric) != [state.bound["semantic_model_name"]]:
                    return FuncToolResult(
                        success=0,
                        error="The bound OSI artifact no longer declares the selected semantic model.",
                        result={"code": "semantic_model_target_invalid"},
                    )
                if not state.touched_metric_names:
                    return FuncToolResult(
                        success=0,
                        error="No metrics were touched in the bound OSI semantic model during this run.",
                    )
                if state.touched_dataset_names or state.target_mutated:
                    semantic_model_files = [metric_file]
                current_metric_names = self.extract_osi_metric_names(abs_bound_metric)
                present_metric_names, absent_metric_names = state.partition_touched_metrics(current_metric_names)
                osi_touched_metric_names = set(state.touched_metric_names)
                # One operation journal is sufficient: final YAML presence
                # decides upsert versus deletion, including same-run reversals.
                osi_metric_names_to_sync = set(present_metric_names)
                osi_absent_metric_names = set(absent_metric_names)
                if not self.generation_evidence.semantic_artifact_validation_passed(
                    state.bound["semantic_model_name"],
                    abs_bound_metric,
                ):
                    return FuncToolResult(
                        success=0,
                        error="validate_semantic must pass for the exact bound OSI semantic-model artifact.",
                    )

            if not exact_osi_target_required and not self.generation_evidence.validation_passed:
                return FuncToolResult(
                    success=0,
                    error=(
                        "validate_semantic must pass before publishing metrics. "
                        "Call validate_semantic, fix any issues, and retry publish_metrics."
                    ),
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                    },
                )

            logger.info(
                f"Metric generation completed: metric_file={metric_file}, "
                f"semantic_model_files={semantic_model_files}, "
                f"metric_sqls={metric_sqls}"
            )

            # Resolve LLM-reported paths against the project's subject/ tree.
            # Reject anything that escapes the per-kind semantic-model sandbox
            # before opening or syncing files.
            from datus.storage.artifact_path import resolve_kb_sandbox_path

            subject_root = str(get_path_manager(agent_config=self.agent_config).subject_dir)

            def _resolve(path: str, kind: str) -> str:
                if not path:
                    return ""
                return resolve_kb_sandbox_path(path, kind, subject_root) or ""

            abs_metric = _resolve(metric_file, "metric")
            if not isinstance(semantic_model_files, list):
                return FuncToolResult(
                    success=0,
                    error="semantic_model_files must be a list of semantic model YAML paths",
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                    },
                )
            abs_semantic_files: List[str] = []
            for candidate_semantic_model_file in semantic_model_files:
                abs_semantic = _resolve(candidate_semantic_model_file, "semantic")
                if not abs_semantic:
                    return FuncToolResult(
                        success=0,
                        error=(
                            "semantic_model_files contains path outside Knowledge Base sandbox: "
                            f"{candidate_semantic_model_file!r}"
                        ),
                        result={
                            "metric_file": metric_file,
                            "semantic_model_files": semantic_model_files,
                            "metric_sqls": metric_sqls,
                        },
                    )
                abs_semantic_files.append(abs_semantic)
            if not abs_metric:
                return FuncToolResult(
                    success=0,
                    error=f"metric_file escapes Knowledge Base sandbox: {metric_file!r}",
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                    },
                )
            if osi_metric_names_to_sync is None:
                osi_metric_names_to_sync = set(self.extract_osi_metric_names(abs_metric))
            sync_kwargs: Dict[str, Any] = {"metric_names_to_sync": osi_metric_names_to_sync}
            if osi_touched_metric_names is not None:
                sync_kwargs["metric_names_to_reconcile"] = osi_touched_metric_names
            sync_result = self._sync_osi_metric_to_db(
                abs_metric,
                abs_semantic_files,
                metric_sqls,
                **sync_kwargs,
            )
            if not sync_result.get("success"):
                return FuncToolResult(
                    success=0,
                    error=f"OSI metric file written but KB sync failed: {sync_result.get('error', 'unknown')}",
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                        "sync": sync_result,
                    },
                )
            kb_sync_metric_names = (
                set(osi_touched_metric_names) if osi_touched_metric_names is not None else set(osi_metric_names_to_sync)
            )
            self.generation_evidence.mark_kb_sync("metric", kb_sync_metric_names)
            if sync_result.get("semantic_synced"):
                self.generation_evidence.mark_kb_sync("semantic")
            if self.osi_target_state is not None:
                self.osi_target_state.clear_artifact_snapshot()
            return FuncToolResult(
                result={
                    "message": "OSI metric generation completed and synced to Knowledge Base",
                    "metric_file": metric_file,
                    "semantic_model_files": semantic_model_files,
                    "metric_sqls": metric_sqls,
                    "deleted_metric_names": sorted(osi_absent_metric_names),
                    "sync": sync_result,
                }
            )

        except Exception as e:
            logger.error(f"Error completing metric generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

    def _resolve_generation_path(self, path: str, kind: str) -> str:
        if not path:
            return ""
        from datus.storage.artifact_path import resolve_kb_sandbox_path

        subject_root = str(get_path_manager(agent_config=self.agent_config).subject_dir)
        return resolve_kb_sandbox_path(path, kind, subject_root) or ""

    @staticmethod
    def _iter_yaml_docs(path: str) -> List[dict]:
        p = Path(path)
        files = sorted(p.rglob("*.yml")) + sorted(p.rglob("*.yaml")) if p.is_dir() else [p]
        docs: List[dict] = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    loaded = list(yaml.safe_load_all(f))
            except (OSError, yaml.YAMLError):
                continue
            docs.extend(doc for doc in loaded if isinstance(doc, dict))
        return docs

    def extract_osi_metric_names(self, metric_path: str) -> List[str]:
        """Return metric names from OSI core documents or compatibility metric docs."""
        names: List[str] = []
        for doc in self._iter_yaml_docs(metric_path):
            semantic_models = doc.get("semantic_model")
            if isinstance(semantic_models, list):
                for model in semantic_models:
                    if not isinstance(model, dict):
                        continue
                    for item in model.get("metrics") or []:
                        if isinstance(item, dict) and isinstance(item.get("name"), str):
                            names.append(item["name"])
            metric = doc.get("metric")
            if isinstance(metric, dict) and isinstance(metric.get("name"), str):
                names.append(metric["name"])
            metrics = doc.get("metrics")
            if isinstance(metrics, list):
                for item in metrics:
                    if isinstance(item, dict) and isinstance(item.get("name"), str):
                        names.append(item["name"])
        return names

    def extract_osi_model_names(self, osi_path: str) -> List[str]:
        """Return semantic-model names declared by one OSI artifact."""
        names: List[str] = []
        for doc in self._iter_yaml_docs(osi_path):
            semantic_models = doc.get("semantic_model")
            if isinstance(semantic_models, list):
                for model in semantic_models:
                    if isinstance(model, dict) and isinstance(model.get("name"), str):
                        names.append(model["name"])
            elif isinstance(semantic_models, dict) and isinstance(semantic_models.get("name"), str):
                names.append(semantic_models["name"])
        return self._dedupe_strings(names)

    def extract_osi_dataset_names(self, semantic_model_path: str) -> List[str]:
        """Return dataset names declared in OSI core semantic-model documents."""
        names: List[str] = []
        for doc in self._iter_yaml_docs(semantic_model_path):
            semantic_models = doc.get("semantic_model")
            if isinstance(semantic_models, list):
                for model in semantic_models:
                    if not isinstance(model, dict):
                        continue
                    for item in model.get("datasets") or []:
                        if isinstance(item, dict) and isinstance(item.get("name"), str):
                            names.append(item["name"])
            datasets = doc.get("datasets")
            if not isinstance(datasets, list):
                continue
            for item in datasets:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    names.append(item["name"])
        return names

    def _load_osi_document(
        self,
        metric_file: Optional[str] = None,
        semantic_model_file: Optional[str] = None,
    ):
        artifact_path = metric_file
        model_names = self.extract_osi_model_names(metric_file) if metric_file else []
        if not model_names and semantic_model_file:
            artifact_path = semantic_model_file
            model_names = self.extract_osi_model_names(semantic_model_file)
        if len(model_names) != 1:
            candidates = ", ".join(model_names) or "<none>"
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message=(
                    f"Exactly one semantic model must be identifiable from the target artifact. Found: {candidates}."
                ),
            )
        from datus.tools.semantic_tools.osi_document import load_osi_document

        return load_osi_document(
            str(artifact_path),
            semantic_model_name=model_names[0],
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, list):
            return [GenerationTools._jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [GenerationTools._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): GenerationTools._jsonable(item) for key, item in value.items()}
        return value

    @classmethod
    def _json_dumps(cls, value: Any) -> str:
        cleaned = cls._jsonable(value)
        if cleaned in (None, "", [], {}):
            return ""
        return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _profile_search_text(cls, *values: Any) -> str:
        parts: List[str] = []
        for value in values:
            cleaned = cls._jsonable(value)
            if cleaned in (None, "", [], {}):
                continue
            if isinstance(cleaned, (dict, list)):
                text = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
            else:
                text = str(cleaned)
            if text and text not in parts:
                parts.append(text)
        return "\n".join(parts)

    @classmethod
    def _osi_dataset_columns(cls, dataset: Any) -> List[dict]:
        columns: List[dict] = []
        primary_keys = getattr(dataset, "primary_key", None) or []
        if isinstance(primary_keys, str):
            primary_keys = [primary_keys]
        for key in primary_keys:
            key_name = str(key)
            if not key_name:
                continue
            columns.append(
                {
                    "name": key_name,
                    "expr": key_name,
                    "role": "primary_key",
                    "type": "identifier",
                    "description": "Primary key",
                }
            )

        time_dimension = getattr(dataset, "time_dimension", None)
        if time_dimension and getattr(time_dimension, "name", None):
            time_name = str(time_dimension.name)
            columns.append(
                {
                    "name": time_name,
                    "expr": getattr(time_dimension, "expr", None) or time_name,
                    "role": "time_dimension",
                    "type": "time",
                    "granularity": getattr(time_dimension, "granularity", "") or "",
                    "description": getattr(time_dimension, "description", "") or "",
                    "ai_context": getattr(time_dimension, "ai_context", None),
                }
            )

        dimension_names = {
            str(getattr(dimension, "name", "") or "") for dimension in getattr(dataset, "dimensions", []) or []
        }
        fields = getattr(dataset, "fields", None)
        for field in fields if fields is not None else getattr(dataset, "dimensions", []):
            field_name = str(getattr(field, "name", "") or "")
            if not field_name or field_name in {column["name"] for column in columns}:
                continue
            columns.append(
                {
                    "name": field_name,
                    "expr": getattr(field, "expr", None) or field_name,
                    "role": "dimension" if field_name in dimension_names else "field",
                    "type": str(getattr(field, "type", "") or ""),
                    "granularity": getattr(field, "granularity", "") or "",
                    "description": getattr(field, "description", "") or "",
                    "ai_context": getattr(field, "ai_context", None),
                }
            )
        return [{key: value for key, value in item.items() if value not in (None, "", [], {})} for item in columns]

    @staticmethod
    def _osi_dataset_binding(dataset: Any) -> tuple[str, str]:
        """Split a dataset's source into (physical table leaf, reusable query).

        Exactly one side is populated. A query-backed dataset gets no table, so
        it can never be picked up by a lookup for a physical table that happens
        to share its name.
        """
        source = getattr(dataset, "source", None)
        table_ref = str(getattr(source, "table", "") or "")
        query = str(getattr(source, "query", "") or "")
        if query:
            return "", query
        return table_ref.split(".")[-1], ""

    @classmethod
    def _osi_dataset_rows(
        cls,
        *,
        doc: Any,
        dataset: Any,
        source_table: str,
        source_query: str,
        db_parts: dict[str, str],
        yaml_path: str,
    ) -> List[dict]:
        """Project one dataset into its own row plus one row per field."""
        semantic_model_name = str(getattr(doc, "name", "") or "")
        dataset_name = str(getattr(dataset, "name", "") or source_table)
        description = getattr(dataset, "description", "") or ""
        ai_context = getattr(dataset, "ai_context", None)
        stamped_at = datetime.now().replace(microsecond=0)

        rows: List[dict] = [
            {
                "id": dataset_row_id(semantic_model_name, dataset_name),
                "kind": KIND_DATASET,
                "semantic_model_name": semantic_model_name,
                "dataset_name": dataset_name,
                "name": dataset_name,
                "source_table": source_table,
                "source_query": source_query,
                "description": description,
                "ai_context_json": cls._json_dumps(ai_context),
                "search_text": cls._profile_search_text(
                    semantic_model_name, dataset_name, source_table, description, ai_context
                ),
                "yaml_path": yaml_path,
                "updated_at": stamped_at,
                **db_parts,
            }
        ]

        for column in cls._osi_dataset_columns(dataset):
            field_name = str(column.get("name") or "")
            if not field_name:
                continue
            role = str(column.get("role") or "")
            rows.append(
                {
                    "id": field_row_id(semantic_model_name, dataset_name, field_name),
                    "kind": KIND_FIELD,
                    "semantic_model_name": semantic_model_name,
                    "dataset_name": dataset_name,
                    "name": field_name,
                    # Carried from the dataset so a lookup can narrow fields to
                    # one physical table without resolving the dataset first.
                    "source_table": source_table,
                    "expr": str(column.get("expr") or field_name),
                    "field_type": str(column.get("type") or ""),
                    "is_primary_key": role == "primary_key",
                    "is_time": role == "time_dimension",
                    "is_dimension": role in ("dimension", "time_dimension"),
                    "time_granularity": str(column.get("granularity") or ""),
                    "description": str(column.get("description") or ""),
                    "ai_context_json": cls._json_dumps(column.get("ai_context")),
                    "search_text": cls._profile_search_text(
                        dataset_name,
                        field_name,
                        column.get("expr"),
                        column.get("description"),
                        column.get("ai_context"),
                    ),
                    "yaml_path": yaml_path,
                    "updated_at": stamped_at,
                    **db_parts,
                }
            )
        return rows

    @classmethod
    def _osi_relationship_rows(cls, *, doc: Any, yaml_path: str) -> List[dict]:
        """Project a model's relationships, which belong to the model itself.

        Endpoints are dataset names, and those names only mean anything inside
        their own semantic model — which is why these rows are never merged
        across models when a table is read back.
        """
        semantic_model_name = str(getattr(doc, "name", "") or "")
        stamped_at = datetime.now().replace(microsecond=0)
        rows: List[dict] = []
        for relationship in getattr(doc, "relationships", []) or []:
            from_dataset = cls._relationship_endpoint(relationship, "from", "from_dataset")
            to_dataset = cls._relationship_endpoint(relationship, "to", "to_dataset")
            from_columns = cls._relationship_columns(relationship, "from_columns", "from_identifier")
            to_columns = cls._relationship_columns(relationship, "to_columns", "to_identifier")
            if not from_dataset and not to_dataset:
                continue
            name = str(getattr(relationship, "name", "") or "")
            # An unnamed relationship still needs a stable identity, or two of
            # them would collapse onto the same row. Its endpoints and joining
            # columns are what make it unique in that case.
            identity = name or f"{from_dataset}->{to_dataset}:{','.join(from_columns)}"
            # The synthesized identity is also what the row is called, so an
            # unnamed relationship still sorts and displays deterministically.
            ai_context = getattr(relationship, "ai_context", None)
            rows.append(
                {
                    "id": relationship_row_id(semantic_model_name, identity),
                    "kind": KIND_RELATIONSHIP,
                    "semantic_model_name": semantic_model_name,
                    "dataset_name": "",
                    "name": identity,
                    "from_dataset": from_dataset,
                    "to_dataset": to_dataset,
                    "from_columns_json": cls._json_dumps(from_columns),
                    "to_columns_json": cls._json_dumps(to_columns),
                    "rel_type": str(getattr(relationship, "type", "") or ""),
                    "join_type": str(getattr(relationship, "join_type", "") or ""),
                    "ai_context_json": cls._json_dumps(ai_context),
                    "search_text": cls._profile_search_text(name, from_dataset, to_dataset, ai_context),
                    "yaml_path": yaml_path,
                    "updated_at": stamped_at,
                }
            )
        return rows

    @staticmethod
    def _current_db_parts(agent_config: AgentConfig) -> dict[str, str]:
        try:
            current_db_config = agent_config.current_db_config()
        except Exception:
            current_db_config = object()
        runtime_db_context_getter = getattr(agent_config, "runtime_db_context", None)
        runtime_db_context = runtime_db_context_getter() if callable(runtime_db_context_getter) else {}
        runtime_db_context = runtime_db_context if isinstance(runtime_db_context, dict) else {}
        return {
            "catalog_name": runtime_db_context.get("catalog")
            or runtime_db_context.get("catalog_name")
            or getattr(current_db_config, "catalog", "")
            or "",
            "database_name": runtime_db_context.get("database")
            or runtime_db_context.get("database_name")
            or getattr(current_db_config, "database", "")
            or "",
            "schema_name": runtime_db_context.get("schema")
            or runtime_db_context.get("db_schema")
            or runtime_db_context.get("schema_name")
            or getattr(current_db_config, "schema", "")
            or "",
        }

    def _dataset_db_parts(self, dataset: Any, default_db_parts: dict[str, str]) -> dict[str, str]:
        """Hierarchy for one dataset: a qualified source table overrides the
        connection defaults, so same-named tables in different databases keep
        distinct storage ids (issue #1084)."""
        from datus.utils.sql_utils import parse_table_name_parts

        source = getattr(dataset, "source", None)
        table_ref = str(getattr(source, "table", "") or "")
        if "." not in table_ref:
            return default_db_parts
        parsed = parse_table_name_parts(table_ref, dialect=self.agent_config.db_type or "snowflake")
        return {
            "catalog_name": parsed.get("catalog_name") or default_db_parts["catalog_name"],
            "database_name": parsed.get("database_name") or default_db_parts["database_name"],
            "schema_name": parsed.get("schema_name") or default_db_parts["schema_name"],
        }

    @staticmethod
    def _dataset_lookup(doc: Any) -> dict[str, Any]:
        return {getattr(dataset, "name", ""): dataset for dataset in getattr(doc, "datasets", [])}

    @staticmethod
    def _metric_subject_path(metric: Any) -> list[str]:
        subject_path = getattr(metric, "subject_path", None)
        if isinstance(subject_path, list) and subject_path:
            return [str(part) for part in subject_path if str(part)]
        dataset = getattr(metric, "dataset", None) or "Unknown"
        return ["Metrics", str(dataset)]

    @staticmethod
    def _metric_expression(metric: Any) -> str:
        expression = getattr(metric, "expression", None)
        if expression:
            return str(expression)
        numerator = getattr(metric, "numerator", None)
        denominator = getattr(metric, "denominator", None)
        if numerator or denominator:
            return f"{numerator or ''} / {denominator or ''}".strip()
        inputs = getattr(metric, "inputs", None) or []
        if inputs:
            return ", ".join(str(getattr(item, "name", item)) for item in inputs)
        return ""

    @staticmethod
    def _dedupe_strings(values: Iterable[Any]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @staticmethod
    def _dataset_primary_keys(dataset: Any) -> List[str]:
        primary_keys = getattr(dataset, "primary_key", None) or []
        if isinstance(primary_keys, str):
            primary_keys = [primary_keys]
        return [str(key) for key in primary_keys if str(key)]

    @staticmethod
    def _relationship_endpoint(relationship: Any, core_name: str, normalized_name: str) -> str:
        return str(getattr(relationship, normalized_name, None) or getattr(relationship, core_name, None) or "")

    @staticmethod
    def _relationship_columns(relationship: Any, core_name: str, normalized_name: str) -> List[str]:
        columns = getattr(relationship, core_name, None)
        if isinstance(columns, str):
            return [columns] if columns else []
        if isinstance(columns, list):
            return [str(column) for column in columns if str(column)]
        normalized_column = str(getattr(relationship, normalized_name, None) or "")
        return [normalized_column] if normalized_column else []

    @staticmethod
    def _relationship_path_name(relationship: Any) -> str:
        """Return the native OSI relationship name used in dimension paths."""
        return str(getattr(relationship, "name", None) or "")

    @classmethod
    def _metric_dataset_names(
        cls,
        metric: Any,
        *,
        metrics_by_name: Dict[str, Any],
        default_dataset: str = "",
        seen_metrics: Optional[set[str]] = None,
    ) -> List[str]:
        dataset = getattr(metric, "dataset", None)
        if dataset:
            return [str(dataset)]
        datasets = getattr(metric, "datasets", None) or []
        if datasets:
            return cls._dedupe_strings(datasets)

        metric_name = str(getattr(metric, "name", "") or "")
        seen_metrics = seen_metrics or set()
        if metric_name in seen_metrics:
            return []
        if metric_name:
            seen_metrics.add(metric_name)

        dataset_names: List[str] = []
        for input_metric in getattr(metric, "inputs", None) or []:
            input_name = str(getattr(input_metric, "name", input_metric) or "")
            referenced = metrics_by_name.get(input_name)
            if referenced is None:
                continue
            dataset_names.extend(
                cls._metric_dataset_names(
                    referenced,
                    metrics_by_name=metrics_by_name,
                    default_dataset=default_dataset,
                    seen_metrics=seen_metrics,
                )
            )

        if dataset_names:
            return cls._dedupe_strings(dataset_names)
        if getattr(metric, "measures", None) and default_dataset:
            return [default_dataset]
        return []

    @classmethod
    def _dataset_dimensions_with_relationships(
        cls,
        doc: Any,
        dataset_name: str,
        *,
        prefix: Optional[List[str]] = None,
        visited: Optional[set[str]] = None,
    ) -> List[str]:
        datasets = cls._dataset_lookup(doc)
        dataset = datasets.get(dataset_name)
        if dataset is None:
            return []

        prefix = prefix or []
        visited = visited or set()
        visited.add(dataset_name)

        dimensions: List[str] = []
        time_dimension = getattr(dataset, "time_dimension", None)
        if time_dimension and getattr(time_dimension, "name", None):
            dimensions.append("__".join([*prefix, str(time_dimension.name)]) if prefix else str(time_dimension.name))
        dimensions.extend(
            "__".join([*prefix, str(dim.name)]) if prefix else str(dim.name)
            for dim in getattr(dataset, "dimensions", [])
            if getattr(dim, "name", None)
        )

        for relationship in getattr(doc, "relationships", []) or []:
            if cls._relationship_endpoint(relationship, "from", "from_dataset") != dataset_name:
                continue
            to_dataset_name = cls._relationship_endpoint(relationship, "to", "to_dataset")
            if not to_dataset_name or to_dataset_name in visited:
                continue
            to_dataset = datasets.get(to_dataset_name)
            if to_dataset is None:
                continue
            relationship_name = cls._relationship_path_name(relationship)
            if not relationship_name:
                logger.warning(
                    "Skipping unnamed OSI relationship %s -> %s; joined dimensions are omitted from the path.",
                    dataset_name,
                    to_dataset_name,
                )
                continue
            dimensions.extend(
                cls._dataset_dimensions_with_relationships(
                    doc,
                    to_dataset_name,
                    prefix=[*prefix, relationship_name],
                    visited=set(visited),
                )
            )
        return cls._dedupe_strings(dimensions)

    @classmethod
    def _metric_query_dimensions(cls, doc: Any, metric: Any) -> List[str]:
        metrics_by_name = {getattr(item, "name", ""): item for item in getattr(doc, "metrics", [])}
        default_dataset = (
            str(getattr(getattr(doc, "datasets", [None])[0], "name", "") or "")
            if getattr(doc, "datasets", None)
            else ""
        )
        dimensions: List[str] = []
        for dataset_name in cls._metric_dataset_names(
            metric,
            metrics_by_name=metrics_by_name,
            default_dataset=default_dataset,
        ):
            dimensions.extend(cls._dataset_dimensions_with_relationships(doc, dataset_name))
        return cls._dedupe_strings(dimensions)

    @classmethod
    def _metric_entities(cls, doc: Any, metric: Any) -> List[str]:
        datasets = cls._dataset_lookup(doc)
        metrics_by_name = {getattr(item, "name", ""): item for item in getattr(doc, "metrics", [])}
        default_dataset = (
            str(getattr(getattr(doc, "datasets", [None])[0], "name", "") or "")
            if getattr(doc, "datasets", None)
            else ""
        )
        entities: List[str] = []
        for dataset_name in cls._metric_dataset_names(
            metric,
            metrics_by_name=metrics_by_name,
            default_dataset=default_dataset,
        ):
            entities.extend(cls._dataset_primary_keys(datasets.get(dataset_name)))
        return cls._dedupe_strings(entities)

    def _sync_osi_semantic_objects_to_db(
        self,
        semantic_model_path: str,
        *,
        doc: Any = None,
        prepare_only: bool = False,
    ) -> dict:
        """Sync OSI datasets into the semantic object store."""
        try:
            # Checked before any work: a caller that prepares rows and then
            # finds no store would otherwise report success having written
            # nothing.
            if self.semantic_dataset_rag is None:
                return {"success": False, "error": "Semantic dataset storage is unavailable"}
            target_dataset_names = set(self.extract_osi_dataset_names(semantic_model_path))
            if not target_dataset_names:
                return {
                    "success": False,
                    "error": f"No OSI datasets found in semantic model file to sync: {semantic_model_path}",
                }

            doc = doc or self._load_osi_document(semantic_model_file=semantic_model_path)
            default_db_parts = self._current_db_parts(self.agent_config)
            dataset_rows: List[dict] = []
            synced_items: List[str] = []

            for dataset in getattr(doc, "datasets", []):
                dataset_name = getattr(dataset, "name", "")
                if dataset_name not in target_dataset_names:
                    continue
                db_parts = self._dataset_db_parts(dataset, default_db_parts)
                yaml_path = semantic_model_path
                source_table, source_query = self._osi_dataset_binding(dataset)
                synced_items.append(source_table or dataset_name)
                dataset_rows.extend(
                    self._osi_dataset_rows(
                        doc=doc,
                        dataset=dataset,
                        source_table=source_table,
                        source_query=source_query,
                        # A query-backed dataset has no place in the warehouse
                        # namespace, so it carries no coordinates either.
                        db_parts=db_parts if source_table else {key: "" for key in db_parts},
                        yaml_path=yaml_path,
                    )
                )

            # Relationships hang off the model, not any single dataset, so they
            # are projected once per document rather than per dataset.
            dataset_rows.extend(self._osi_relationship_rows(doc=doc, yaml_path=semantic_model_path))

            if not dataset_rows:
                return {
                    "success": False,
                    "error": (
                        "OSI datasets declared in semantic model file were not found after loading datasource "
                        f"context: {', '.join(sorted(target_dataset_names))}"
                    ),
                }
            if prepare_only:
                return {
                    "success": True,
                    "semantic_dataset_rows": dataset_rows,
                    "synced_items": synced_items,
                }
            replacement_plans = [(self.semantic_dataset_rag, semantic_model_path, dataset_rows)]
            snapshots = snapshot_artifact_replacements(replacement_plans)
            try:
                self.semantic_dataset_rag.upsert_batch(dataset_rows)
                self.semantic_dataset_rag.create_indices()
                profile_count = len(dataset_rows)
                delete_stale_artifact_rows(replacement_plans)
            except Exception as sync_exc:
                restore_failures = restore_artifact_replacements(snapshots)
                if restore_failures:
                    raise RuntimeError(
                        "OSI semantic replacement failed and rollback was incomplete for: "
                        f"{', '.join(restore_failures)}"
                    ) from sync_exc
                raise
            return {
                "success": True,
                "message": f"Synced {profile_count} OSI semantic row(s): {', '.join(synced_items[:5])}",
                "semantic_dataset_rows": profile_count,
            }
        except Exception as e:
            logger.error(f"Error syncing OSI semantic objects to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _build_osi_metric_objects(
        self,
        *,
        doc: Any,
        metric_file: str,
        target_metric_names: set[str],
        metric_sqls: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        """Materialize metric rows from raw presentation and compiled semantics."""

        semantic_model_name = str(getattr(doc, "name", "") or "")
        db_parts = self._current_db_parts(self.agent_config)
        compiled_catalog = self._compiled_metric_catalog(metric_file) if target_metric_names else None
        if compiled_catalog is not None:
            missing = target_metric_names.difference(compiled_catalog)
            if missing:
                raise ValueError(
                    f"Configured semantic adapter did not compile target metric(s): {', '.join(sorted(missing))}"
                )
        metric_objects: List[dict] = []
        for metric in getattr(doc, "metrics", []):
            metric_name = getattr(metric, "name", "")
            if not metric_name or metric_name not in target_metric_names:
                continue
            compiled = compiled_catalog.get(metric_name) if compiled_catalog is not None else None
            semantic_metric = copy(metric)
            if compiled is not None:
                metadata = getattr(compiled, "metadata", None) or {}
                compiled_datasets = self._dedupe_strings(metadata.get("datasets") or [])
                semantic_metric.datasets = compiled_datasets
                semantic_metric.dataset = compiled_datasets[0] if len(compiled_datasets) == 1 else None

            dimensions = (
                self._dedupe_strings(getattr(compiled, "dimensions", None) or [])
                if compiled is not None
                else self._metric_query_dimensions(doc, semantic_metric)
            )
            entities = self._metric_entities(doc, semantic_metric)
            subject_path = self._metric_subject_path(semantic_metric)
            measure_expr = self._metric_expression(metric)
            window_payload = getattr(metric, "window", None)
            metric_type = getattr(compiled, "type", None) if compiled is not None else None
            base_measures = (
                self._dedupe_strings(getattr(compiled, "measures", None) or [])
                if compiled is not None
                else ([measure_expr] if measure_expr else [])
            )
            metric_objects.append(
                {
                    "name": metric_name,
                    "subject_path": subject_path,
                    "semantic_model_name": semantic_model_name,
                    "id": build_metric_id(subject_path, metric_name),
                    "description": getattr(metric, "description", "") or "",
                    "metric_type": (
                        metric_type
                        or ("window" if isinstance(window_payload, dict) else getattr(metric, "kind", None))
                        or "aggregate"
                    ),
                    "measure_expr": measure_expr,
                    "base_measures": base_measures,
                    "dimensions": dimensions,
                    "entities": entities,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": datetime.now().replace(microsecond=0),
                    **db_parts,
                    "sql": metric_sqls.get(metric_name, "") if metric_sqls else "",
                    "yaml_path": metric_file,
                }
            )
        return metric_objects

    def _compiled_metric_catalog(self, metric_file: str) -> Optional[Dict[str, Any]]:
        """Compile one publication artifact through the configured adapter.

        ``None`` means this lightweight object was constructed without a real
        ``AgentConfig`` (primarily isolated unit tests). A configured adapter is
        authoritative: load or pagination failures abort publication instead of
        silently falling back to guesses from raw YAML.
        """

        resolver = getattr(self.agent_config, "resolve_semantic_adapter", None)
        builder = getattr(self.agent_config, "build_semantic_adapter_config", None)
        if not callable(resolver) or not callable(builder):
            return None
        adapter_name = resolver(None)
        if not isinstance(adapter_name, str) or not adapter_name.strip():
            return None

        from datus.tools.semantic_tools.config import SemanticAdapterConfig
        from datus.tools.semantic_tools.paging import metric_catalog_paging
        from datus.tools.semantic_tools.registry import semantic_adapter_registry
        from datus.utils.async_utils import run_async

        adapter_name = adapter_name.strip().lower()
        metadata = semantic_adapter_registry.get_metadata(adapter_name)
        adapter_config = builder(adapter_name)
        config_class = metadata.config_class if metadata and metadata.config_class else SemanticAdapterConfig
        config_fields = getattr(config_class, "model_fields", {})
        artifact_overrides: Dict[str, str] = {}
        if "semantic_model_path" in config_fields:
            artifact_overrides["semantic_model_path"] = metric_file
        if "semantic_models_path" in config_fields:
            artifact_overrides["semantic_models_path"] = str(Path(metric_file).parent)

        if adapter_config is None:
            adapter_config = config_class(**artifact_overrides)
        elif isinstance(adapter_config, dict):
            config_payload = {**adapter_config, **artifact_overrides}
            adapter_config = config_class(**config_payload)
        elif artifact_overrides:
            model_copy = getattr(adapter_config, "model_copy", None)
            if callable(model_copy):
                adapter_config = model_copy(update=artifact_overrides)
            else:
                adapter_config = copy(adapter_config)
                for key, value in artifact_overrides.items():
                    setattr(adapter_config, key, value)

        adapter = semantic_adapter_registry.create_adapter(adapter_name, adapter_config)
        catalog: Dict[str, Any] = {}
        page_size, max_pages = metric_catalog_paging(self.agent_config, adapter_name)
        offset = 0
        for _ in range(max_pages):
            page = list(run_async(adapter.list_metrics(limit=page_size, offset=offset)))
            for metric in page:
                name = str(getattr(metric, "name", "") or "").strip()
                if name:
                    catalog[name] = metric
            if len(page) < page_size:
                return catalog
            offset += len(page)
        if not list(run_async(adapter.list_metrics(limit=page_size, offset=offset))):
            return catalog
        raise ValueError(
            f"Semantic adapter metric catalog exceeds the {page_size * max_pages} metric publication limit"
        )

    @staticmethod
    def _preserve_existing_metric_sql(metric_objects: List[dict], existing_rows: Any) -> None:
        """Keep generated SQL for unchanged metrics during a YAML-only refresh."""

        existing_by_name = {
            normalize_metric_name(row.get("name")): row
            for row in _rows_to_dicts(existing_rows)
            if normalize_metric_name(row.get("name")) and str(row.get("sql") or "").strip()
        }
        for metric_object in metric_objects:
            if str(metric_object.get("sql") or "").strip():
                continue
            existing = existing_by_name.get(normalize_metric_name(metric_object.get("name")))
            if existing is not None and metric_definition_conflict(existing, metric_object) is None:
                metric_object["sql"] = existing["sql"]

    def _sync_osi_metric_to_db(
        self,
        metric_file: str,
        semantic_model_file: Optional[str | List[str]] = None,
        metric_sqls: Optional[Dict[str, str]] = None,
        metric_names_to_sync: Optional[set[str]] = None,
        metric_names_to_reconcile: Optional[set[str]] = None,
    ) -> dict:
        """Upsert the requested scope and reconcile stale rows from final YAML."""
        try:
            full_artifact_sync = metric_names_to_sync is None
            semantic_model_files = (
                list(semantic_model_file)
                if isinstance(semantic_model_file, list)
                else ([semantic_model_file] if semantic_model_file else [])
            )
            declared_metric_names = set(self.extract_osi_metric_names(metric_file))
            declared_metric_names_by_key = {
                normalize_metric_name(name): name for name in declared_metric_names if normalize_metric_name(name)
            }
            reconcile_metric_names = {
                str(name).strip() for name in metric_names_to_reconcile or set() if str(name).strip()
            }
            absent_metric_names = {
                name
                for name in reconcile_metric_names
                if normalize_metric_name(name) not in declared_metric_names_by_key
            }
            target_metric_names = (
                declared_metric_names
                if full_artifact_sync
                else {
                    declared_metric_names_by_key[normalize_metric_name(name)]
                    for name in metric_names_to_sync
                    if normalize_metric_name(name) in declared_metric_names_by_key
                }
            )
            missing_metric_names = {
                str(name).strip()
                for name in metric_names_to_sync or set()
                if str(name).strip() and normalize_metric_name(name) not in declared_metric_names_by_key
            }
            if missing_metric_names:
                return {
                    "success": False,
                    "error": (
                        "OSI metric publish scope contains names not declared in the metric file: "
                        f"{', '.join(sorted(missing_metric_names))}"
                    ),
                }
            if not full_artifact_sync and not target_metric_names and not reconcile_metric_names:
                return {"success": False, "error": "OSI metric publish scope must not be empty"}

            doc = self._load_osi_document(
                metric_file=metric_file,
                semantic_model_file=semantic_model_files[0] if semantic_model_files else None,
            )
            metric_objects = self._build_osi_metric_objects(
                doc=doc,
                metric_file=metric_file,
                target_metric_names=target_metric_names,
                metric_sqls=metric_sqls,
            )

            if target_metric_names and not metric_objects:
                return {
                    "success": False,
                    "error": (
                        "OSI metrics declared in metric file were not found after loading datasource context: "
                        f"{', '.join(sorted(target_metric_names))}"
                    ),
                }

            semantic_replacements: List[tuple[str, List[dict]]] = []
            semantic_replacement_plans = []
            synced_semantic_files: List[str] = []
            for current_semantic_file in semantic_model_files:
                sem_result = self._sync_osi_semantic_objects_to_db(
                    current_semantic_file,
                    prepare_only=True,
                )
                if not sem_result.get("success"):
                    return sem_result
                dataset_rows = list(sem_result.get("semantic_dataset_rows") or [])
                semantic_replacements.append((current_semantic_file, dataset_rows))
                if self.semantic_dataset_rag is not None:
                    semantic_replacement_plans.append((self.semantic_dataset_rag, current_semantic_file, dataset_rows))
                synced_semantic_files.append(current_semantic_file)

            metric_plan = (self.metric_rag, metric_file, metric_objects)
            replacement_plans = [*semantic_replacement_plans, metric_plan]
            snapshots = snapshot_artifact_replacements(replacement_plans)
            metric_snapshot = next(
                (rows for rag, path, rows in snapshots if rag is self.metric_rag and path == metric_file),
                [],
            )
            self._preserve_existing_metric_sql(metric_objects, metric_snapshot)
            if full_artifact_sync:
                for row in _rows_to_dicts(metric_snapshot):
                    previous_name = str(row.get("name") or "").strip()
                    if previous_name and normalize_metric_name(previous_name) not in declared_metric_names_by_key:
                        absent_metric_names.add(previous_name)
            keep_metric_ids = [build_metric_id([], name) for name in declared_metric_names]
            try:
                for _semantic_file, dataset_rows in semantic_replacements:
                    if dataset_rows and self.semantic_dataset_rag is not None:
                        self.semantic_dataset_rag.upsert_batch(dataset_rows)
                        self.semantic_dataset_rag.create_indices()
                if metric_objects:
                    self.metric_rag.upsert_batch(metric_objects)
                    self.metric_rag.create_indices()
                delete_stale_artifact_rows(semantic_replacement_plans)
                self.metric_rag.delete_artifact_rows_except(metric_file, keep_metric_ids)
            except Exception as sync_exc:
                restore_failures = restore_artifact_replacements(snapshots)
                if restore_failures:
                    raise RuntimeError(
                        "OSI semantic and metric replacement failed and rollback was incomplete for: "
                        f"{', '.join(restore_failures)}"
                    ) from sync_exc
                raise
            return {
                "success": True,
                "message": (
                    f"Synced {len(metric_objects)} and reconciled {len(absent_metric_names)} absent OSI metric(s)"
                ),
                "metric_artifact_ids": [obj["id"] for obj in metric_objects],
                "metric_names": [obj["name"] for obj in metric_objects],
                "deleted_metric_names": sorted(absent_metric_names),
                "semantic_synced": bool(synced_semantic_files),
                "semantic_model_files_synced": synced_semantic_files,
            }
        except Exception as e:
            logger.error(f"Error syncing OSI metrics to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def sync_osi_to_db(
        self,
        osi_file_path: str,
        *,
        include_semantic_objects: bool = True,
        include_metrics: bool = True,
    ) -> dict:
        """Sync an explicitly scoped OSI document into the Knowledge Base.

        This is the low-level exact-artifact import/refresh entry. Standalone
        generation nodes reach it through their publish contracts; the unified
        semantic-modeling host finalizer calls it after exact-target validation.

        Requested stores are prepared from one normalized document and replaced
        under one snapshot/restore boundary. Empty metric collections therefore
        remove stale rows, and a later-store failure restores earlier stores.
        The result always carries a ``synced`` count for uniform accounting.
        """
        try:
            if not include_semantic_objects and not include_metrics:
                return {"success": False, "error": "At least one OSI sync scope must be enabled", "synced": 0}
            doc = self._load_osi_document(
                metric_file=osi_file_path if include_metrics else None,
                semantic_model_file=osi_file_path if include_semantic_objects else None,
            )

            replacement_plans = []
            dataset_rows: List[dict] = []
            synced_items: List[str] = []
            if include_semantic_objects:
                prepared_semantic = self._sync_osi_semantic_objects_to_db(
                    osi_file_path,
                    doc=doc,
                    prepare_only=True,
                )
                if not prepared_semantic.get("success"):
                    return {**prepared_semantic, "synced": 0}
                dataset_rows = list(prepared_semantic.get("semantic_dataset_rows") or [])
                synced_items = list(prepared_semantic.get("synced_items") or [])
                if self.semantic_dataset_rag is not None:
                    replacement_plans.append((self.semantic_dataset_rag, osi_file_path, dataset_rows))

            declared_metric_names = set(self.extract_osi_metric_names(osi_file_path))
            metric_objects: List[dict] = []
            if include_metrics:
                metric_objects = self._build_osi_metric_objects(
                    doc=doc,
                    metric_file=osi_file_path,
                    target_metric_names=declared_metric_names,
                )
                if declared_metric_names and not metric_objects:
                    return {
                        "success": False,
                        "error": (
                            "OSI metrics declared in the document were not found after loading datasource context: "
                            f"{', '.join(sorted(declared_metric_names))}"
                        ),
                        "synced": 0,
                    }
                replacement_plans.append((self.metric_rag, osi_file_path, metric_objects))

            snapshots = snapshot_artifact_replacements(replacement_plans)
            deleted_metric_names: set[str] = set()
            if include_metrics:
                metric_snapshot = next(
                    (rows for rag, _path, rows in snapshots if rag is self.metric_rag),
                    [],
                )
                self._preserve_existing_metric_sql(metric_objects, metric_snapshot)
                declared_by_key = {
                    normalize_metric_name(name) for name in declared_metric_names if normalize_metric_name(name)
                }
                for row in _rows_to_dicts(metric_snapshot):
                    previous_name = str(row.get("name") or "").strip()
                    if previous_name and normalize_metric_name(previous_name) not in declared_by_key:
                        deleted_metric_names.add(previous_name)

            try:
                if dataset_rows and self.semantic_dataset_rag is not None:
                    self.semantic_dataset_rag.upsert_batch(dataset_rows)
                    self.semantic_dataset_rag.create_indices()
                if metric_objects:
                    self.metric_rag.upsert_batch(metric_objects)
                    self.metric_rag.create_indices()
                delete_stale_artifact_rows(replacement_plans)
            except Exception as sync_exc:
                restore_failures = restore_artifact_replacements(snapshots)
                if restore_failures:
                    raise RuntimeError(
                        "OSI document replacement failed and rollback was incomplete for: "
                        f"{', '.join(restore_failures)}"
                    ) from sync_exc
                raise

            synced = len(metric_objects) if include_metrics else len(dataset_rows)
            return {
                "success": True,
                "message": (f"Synced {len(dataset_rows)} OSI semantic row(s) and {len(metric_objects)} metric(s)"),
                "semantic_dataset_rows": len(dataset_rows) if self.semantic_dataset_rag is not None else 0,
                "metric_artifact_ids": [obj["id"] for obj in metric_objects],
                "metric_names": [obj["name"] for obj in metric_objects],
                "deleted_metric_names": sorted(deleted_metric_names),
                "semantic_items": synced_items[:5],
                "synced": synced,
            }
        except Exception as e:
            logger.error(f"Error syncing OSI document to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e), "synced": 0}

    def generate_sql_summary_id(self, sql_query: str, comment: str = "") -> FuncToolResult:
        """
        Generate a unique ID for SQL summary based on SQL query and comment.
        """
        try:
            from datus.storage.reference_sql.init_utils import gen_reference_sql_id

            # Generate the ID using the same utility as the storage system
            generated_id = gen_reference_sql_id(sql_query)

            logger.info(f"Generated reference SQL ID: {generated_id}")
            return FuncToolResult(result=generated_id)

        except Exception as e:
            logger.error(f"Error generating reference SQL ID: {e}")
            return FuncToolResult(success=0, error=f"Failed to generate ID: {str(e)}")
