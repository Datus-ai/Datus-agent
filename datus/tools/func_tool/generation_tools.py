# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import json
import os
import re
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import yaml
from agents import Tool
from datus_storage_base.conditions import And, eq
from pydantic import BaseModel, Field

from datus.configuration.agent_config import AgentConfig
from datus.storage.artifact_replacement import (
    delete_stale_artifact_rows,
    restore_artifact_replacements,
    snapshot_artifact_replacements,
)
from datus.storage.metric.store import MetricRAG, build_metric_id, metric_definition_conflict, normalize_metric_name
from datus.storage.semantic_model.store import SemanticModelRAG, _identifier_variants, _normalized_identifier
from datus.storage.table_semantic_profile.store import TableSemanticProfileRAG
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.metric_queryability import summarize_queryability_contracts
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState


class MetricOutputBinding(BaseModel):
    """Bind one request-local SQL output to its published metric name."""

    output_id: str = Field(..., description="Output identity returned by prepare_sql_modeling_plan")
    metric_name: str = Field(..., description="Unique metric name published for that output")


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


def _rag_scope_conditions(rag: Any) -> List[Any]:
    method = getattr(rag, "_sub_agent_conditions", None)
    if not callable(method):
        return []
    try:
        conditions = method()
    except Exception:
        return []
    return conditions if isinstance(conditions, list) else []


def _normalized_semantic_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _canonical_sql_expression(value: Any, dialect: str = "") -> str:
    """Normalize one scalar SQL expression without changing its semantics."""
    expression = str(value or "").strip()
    if not expression:
        return ""
    try:
        import sqlglot
        from sqlglot import expressions as exp

        from datus.utils.sql_utils import parse_read_dialect

        read_dialect = parse_read_dialect(dialect) if dialect else None
        parsed = sqlglot.parse_one(f"SELECT {expression}", read=read_dialect)
        projection = parsed.expressions[0]
        if isinstance(projection, exp.Alias):
            projection = projection.this
        for column in projection.find_all(exp.Column):
            column.set("catalog", None)
            column.set("db", None)
            column.set("table", None)
            if isinstance(column.this, exp.Identifier):
                column.this.set("quoted", False)
        return re.sub(r"\s+", "", projection.sql(comments=False, normalize=True)).lower()
    except Exception:
        return re.sub(r"\s+", "", expression).lower()


def _osi_dimension_fields(path: str | Path) -> List[Dict[str, Any]]:
    """Read executable OSI dimension names and dialect expressions."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = document.get("semantic_model") if isinstance(document, dict) else None
    fields: List[Dict[str, Any]] = []
    for model in models if isinstance(models, list) else []:
        if not isinstance(model, dict):
            continue
        datasets = model.get("datasets")
        for dataset in datasets if isinstance(datasets, list) else []:
            if not isinstance(dataset, dict):
                continue
            dataset_fields = dataset.get("fields")
            for field in dataset_fields if isinstance(dataset_fields, list) else []:
                if not isinstance(field, dict) or "dimension" not in field:
                    continue
                raw_expression = field.get("expression")
                expressions: List[str] = []
                if isinstance(raw_expression, str):
                    expressions.append(raw_expression)
                elif isinstance(raw_expression, dict):
                    dialects = raw_expression.get("dialects")
                    for dialect_expression in dialects if isinstance(dialects, list) else []:
                        if not isinstance(dialect_expression, dict):
                            continue
                        expression = dialect_expression.get("expression")
                        if isinstance(expression, str) and expression.strip():
                            expressions.append(expression)
                fields.append(
                    {
                        "name": str(field.get("name") or "").strip(),
                        "expressions": expressions,
                    }
                )
    return fields


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
        sql_modeling_plan_required: Optional[Callable[[], bool]] = None,
    ):
        self.agent_config = agent_config
        self.generation_evidence = generation_evidence or GenerationEvidence()
        if authoring_format:
            self.authoring_format = str(authoring_format).strip().lower()
        else:
            from datus.agent.node.semantic_authoring import resolve_authoring_format

            self.authoring_format = resolve_authoring_format(agent_config)
        self.metric_rag = MetricRAG(agent_config)
        self.semantic_rag = SemanticModelRAG(agent_config)
        self.table_semantic_profile_rag = None
        if isinstance(getattr(agent_config, "project_name", ""), str):
            try:
                self.table_semantic_profile_rag = TableSemanticProfileRAG(agent_config)
            except Exception as exc:
                logger.debug(f"Failed to initialize table semantic profile storage: {exc}")
        self._semantic_object_exists_cache: Dict[tuple[str, str, str], FuncToolResult] = {}
        self._semantic_table_object_index: Optional[Dict[str, Dict[str, object]]] = None
        self.osi_target_state = osi_target_state
        self.require_bound_osi_target = require_bound_osi_target
        self.sql_modeling_plan_required = sql_modeling_plan_required

    def _require_sql_modeling_plan_if_needed(self) -> None:
        """Enforce SQL preflight only for requests that actually contain SQL."""
        if self.sql_modeling_plan_required is not None and self.sql_modeling_plan_required():
            self.generation_evidence.require_sql_modeling_plan()

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
                # For column, use vector search + post-filter
                storage = self.semantic_rag.storage
                results = storage.search_objects(
                    query_text=object_name,
                    kinds=[normalized_kind],
                    table_name=table_context if table_context else None,
                    top_n=5,
                    extra_conditions=_rag_scope_conditions(self.semantic_rag),
                )
                # Determine target table from explicit context or dotted name
                target_table = None
                if table_context:
                    target_table = table_context.lower()
                elif "." in object_name:
                    target_table = object_name.rsplit(".", 1)[0].lower()

                for obj in _rows_to_dicts(results):
                    name_match = obj.get("name", "").lower() == target_name
                    if target_table:
                        table_match = obj.get("table_name", "").lower() == target_table
                        if name_match and table_match:
                            found_object = obj
                            break
                    elif name_match:
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

        storage = self.semantic_rag.storage
        select_fields = ["id", "name", "kind", "table_name", "fq_name"]
        rows = storage.search_all(
            where=And([eq("kind", "table")] + _rag_scope_conditions(self.semantic_rag)),
            select_fields=select_fields,
        )
        index: Dict[str, Dict[str, object]] = {}
        for obj in _rows_to_dicts(rows):
            for field in ("name", "table_name", "fq_name"):
                value = str(obj.get(field) or "").strip()
                if not value:
                    continue
                for variant in _identifier_variants(value):
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
            self._require_sql_modeling_plan_if_needed()
            if not self._is_osi_authoring() and not semantic_model_files:
                return FuncToolResult(
                    success=0,
                    error="semantic_model_files must contain at least one generated semantic model YAML path.",
                )
            osi_target: Optional[tuple[str, str]] = None
            if self._is_osi_authoring():
                semantic_model_file, resolved, model_name = self.resolve_planned_osi_semantic_target()
                semantic_model_files = [semantic_model_file]
                osi_target = (resolved, model_name)
                validation_passed = self.generation_evidence.semantic_artifact_validation_passed(model_name, resolved)
            else:
                validation_passed = self.generation_evidence.validation_passed

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

            if self._is_osi_authoring():
                assert osi_target is not None
                resolved, _model_name = osi_target
                query_backed_error = self._validate_required_query_backed_sources(
                    [resolved],
                    include_active_root=False,
                )
                if query_backed_error:
                    return FuncToolResult(
                        success=0,
                        error=query_backed_error,
                        result={"semantic_model_files": semantic_model_files},
                    )
                dimension_error, missing_dimensions = self._validate_required_osi_group_by_dimensions(resolved)
                if dimension_error:
                    return FuncToolResult(
                        success=0,
                        error=dimension_error,
                        result={
                            "semantic_model_files": semantic_model_files,
                            "missing_group_by_dimensions": missing_dimensions,
                        },
                    )
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
                return FuncToolResult(
                    result={
                        "message": f"Semantic model generation completed and synced {len(sync_results)} OSI file(s)",
                        "semantic_model_files": semantic_model_files,
                        "sync": sync_results,
                    }
                )

            from datus.cli.generation_hooks import GenerationHooks, resolve_kb_sandbox_path

            subject_root = str(get_path_manager(agent_config=self.agent_config).subject_dir)
            resolved_semantic_files = []
            for semantic_model_file in semantic_model_files:
                resolved = resolve_kb_sandbox_path(semantic_model_file, "semantic", subject_root)
                if not resolved:
                    return FuncToolResult(
                        success=0,
                        error=f"semantic_model_file escapes Knowledge Base sandbox: {semantic_model_file!r}",
                        result={"semantic_model_files": semantic_model_files},
                    )
                resolved_semantic_files.append(resolved)
            query_backed_error = self._validate_required_query_backed_sources(resolved_semantic_files)
            if query_backed_error:
                return FuncToolResult(
                    success=0,
                    error=query_backed_error,
                    result={"semantic_model_files": semantic_model_files},
                )

            sync_results = []
            for resolved in resolved_semantic_files:
                sync_result = GenerationHooks._sync_semantic_to_db(
                    resolved,
                    self.agent_config,
                    include_semantic_objects=True,
                    include_metrics=False,
                )
                sync_results.append(sync_result)
                if not sync_result.get("success"):
                    return FuncToolResult(
                        success=0,
                        error=f"Semantic model KB sync failed: {sync_result.get('error', 'unknown')}",
                        result={"semantic_model_files": semantic_model_files, "sync": sync_results},
                    )
            self.generation_evidence.mark_kb_sync("semantic")
            return FuncToolResult(
                result={
                    "message": f"Published {len(semantic_model_files)} semantic model file(s)",
                    "semantic_model_files": semantic_model_files,
                    "sync": sync_results,
                }
            )

        except Exception as e:
            logger.error(f"Error completing semantic model generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

    def publish_metrics(
        self,
        metric_file: str,
        metric_output_bindings: Optional[List[MetricOutputBinding]] = None,
    ) -> FuncToolResult:
        """Validate publication evidence and sync metric artifacts to storage.

        Args:
            metric_file: Path to the generated metric YAML file (required).
                Relative paths (e.g. ``"metrics/orders_metrics.yml"``) are preferred
                and resolved against the sub-agent's semantic-model workspace using
                the live ``agent_config.current_datasource``. Absolute paths are only
                accepted when they resolve inside the Knowledge Base semantic-model
                sandbox.
            metric_output_bindings: Bind every request-local
                ``output_id`` from the SQL modeling plan to its published metric
                name. Required only when the request has planned metric outputs.

        Returns:
            dict: Result containing completion message, file paths, metric SQLs, and sync status
        """
        try:
            self._require_sql_modeling_plan_if_needed()
            metric_sqls = dict(self.generation_evidence.metric_sqls)
            # OSI gen_metrics owns only the metrics collection. Semantic
            # objects are authored and synced by gen_semantic_model.
            semantic_model_files = (
                [] if self._is_osi_authoring() else self.generation_evidence.semantic_model_mutations(metric_file)
            )

            exact_osi_target_required = self._is_osi_authoring() and self.require_bound_osi_target
            osi_metric_names_to_sync: Optional[set[str]] = None
            metric_output_bindings = [
                binding.model_dump() if isinstance(binding, MetricOutputBinding) else dict(binding)
                for binding in metric_output_bindings or []
            ]
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
                if not state.authored_metric_names:
                    return FuncToolResult(
                        success=0,
                        error="No metrics were authored in the bound OSI semantic model during this run.",
                    )
                osi_metric_names_to_sync = set(state.authored_metric_names)
                current_metric_names = set(self.extract_osi_metric_names(abs_bound_metric))
                missing_authored_metrics = [
                    name for name in state.authored_metric_names if name not in current_metric_names
                ]
                if missing_authored_metrics:
                    return FuncToolResult(
                        success=0,
                        error=(
                            "The bound OSI artifact no longer contains every metric authored in this run: "
                            f"{', '.join(missing_authored_metrics)}."
                        ),
                        result={"code": "semantic_model_target_invalid"},
                    )
                binding_error, metric_output_bindings = self._validate_metric_output_bindings(
                    metric_output_bindings,
                    osi_metric_names_to_sync,
                )
                if binding_error:
                    return FuncToolResult(
                        success=0,
                        error=binding_error,
                        result={
                            "metric_file": metric_file,
                            "metric_output_bindings": metric_output_bindings,
                        },
                    )
                self.generation_evidence.bind_metric_output_names(metric_output_bindings)
                if not self.generation_evidence.semantic_artifact_validation_passed(
                    state.bound["semantic_model_name"],
                    abs_bound_metric,
                ):
                    return FuncToolResult(
                        success=0,
                        error="validate_semantic must pass for the exact bound OSI semantic-model artifact.",
                    )
                if not self.generation_evidence.has_metric_dry_run(state.authored_metric_names):
                    return FuncToolResult(
                        success=0,
                        error=(
                            "query_metrics(dry_run=True) must pass for every metric authored "
                            "in the bound OSI semantic model."
                        ),
                    )
                if not self.generation_evidence.has_required_queryability_dry_runs(state.authored_metric_names):
                    missing_contracts = self.generation_evidence.missing_queryability_contracts(
                        state.authored_metric_names
                    )
                    return FuncToolResult(
                        success=0,
                        error=(
                            "query_metrics(dry_run=True) must pass with the source SQL group-by dimensions "
                            "before publishing metrics. Run a dry-run query for the authored metric names "
                            "with the matching dimensions/time grain, fix semantic model join or dimension "
                            f"issues, and retry. Missing: {summarize_queryability_contracts(missing_contracts)}"
                        ),
                        result={"queryability_contracts": missing_contracts},
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

            if not exact_osi_target_required and not self.generation_evidence.metric_dry_run_passed:
                return FuncToolResult(
                    success=0,
                    error=(
                        "query_metrics(dry_run=True) must pass before publishing metrics. "
                        "Run a dry-run query for the generated metric(s), fix any issues, and retry "
                        "publish_metrics."
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
            from datus.cli.generation_hooks import resolve_kb_sandbox_path

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
            query_backed_error = self._validate_required_query_backed_sources([abs_metric, *abs_semantic_files])
            if query_backed_error:
                return FuncToolResult(
                    success=0,
                    error=query_backed_error,
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                    },
                )

            if self._is_osi_authoring():
                if osi_metric_names_to_sync is None:
                    osi_metric_names_to_sync = set(self.extract_osi_metric_names(abs_metric))
                    binding_error, metric_output_bindings = self._validate_metric_output_bindings(
                        metric_output_bindings,
                        osi_metric_names_to_sync,
                    )
                    if binding_error:
                        return FuncToolResult(
                            success=0,
                            error=binding_error,
                            result={
                                "metric_file": metric_file,
                                "metric_output_bindings": metric_output_bindings,
                            },
                        )
                    self.generation_evidence.bind_metric_output_names(metric_output_bindings)
                sync_result = self._sync_osi_metric_to_db(
                    abs_metric,
                    abs_semantic_files,
                    metric_sqls,
                    metric_names_to_sync=osi_metric_names_to_sync,
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
                self.generation_evidence.mark_kb_sync("metric", osi_metric_names_to_sync)
                if sync_result.get("semantic_synced"):
                    self.generation_evidence.mark_kb_sync("semantic")
                if self.osi_target_state is not None:
                    self.osi_target_state.clear_metric_snapshot()
                return FuncToolResult(
                    result={
                        "message": "OSI metric generation completed and synced to Knowledge Base",
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                        "metric_output_bindings": metric_output_bindings,
                        "sync": sync_result,
                    }
                )

            # Pre-flight: refuse to sync a metric file that has no `metric:`
            # YAML blocks. The LLM occasionally fills the file with markdown
            # documentation and relies on `create_metric: true` measures
            # (which never reach the KB vector DB). Catching it here gives a
            # crisp, actionable error the LLM can act on without a roundtrip
            # through the deeper sync path.
            preflight_error = self._validate_metric_file_has_blocks(abs_metric)
            if preflight_error:
                return FuncToolResult(
                    success=0,
                    error=preflight_error,
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                    },
                )
            metric_names = self._extract_metric_names_from_file(abs_metric)
            metric_definitions = self._extract_metric_definitions_from_file(abs_metric)
            metric_names_to_sync = self._metric_names_to_sync(metric_names, metric_sqls)
            scoped_metric_names = self._filter_metric_names(metric_names, metric_names_to_sync)
            scoped_metric_definitions = self._filter_metric_definitions(metric_definitions, metric_names_to_sync)
            binding_error, metric_output_bindings = self._validate_metric_output_bindings(
                metric_output_bindings,
                scoped_metric_names,
            )
            if binding_error:
                return FuncToolResult(
                    success=0,
                    error=binding_error,
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                        "metric_output_bindings": metric_output_bindings,
                    },
                )
            self.generation_evidence.bind_metric_output_names(metric_output_bindings)
            if metric_names_to_sync is not None and not scoped_metric_definitions:
                return FuncToolResult(
                    success=0,
                    error=(
                        "The recorded dry-run SQL does not reference any metric declared in metric_file. "
                        "Run query_metrics(dry_run=True) for the metric names being published."
                    ),
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                        "metric_output_bindings": metric_output_bindings,
                    },
                )
            conflict_error = self._validate_metric_name_conflicts(scoped_metric_definitions)
            if conflict_error:
                return FuncToolResult(
                    success=0,
                    error=conflict_error,
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                    },
                )
            required_metric_names = self._metric_names_requiring_dry_run(
                scoped_metric_names, scoped_metric_definitions, metric_sqls
            )
            if required_metric_names and not self.generation_evidence.has_metric_dry_run(required_metric_names):
                return FuncToolResult(
                    success=0,
                    error=(
                        "query_metrics(dry_run=True) must pass for generated metric(s): "
                        f"{', '.join(required_metric_names)}. Run a dry-run query for these metric names, "
                        "fix any issues, and retry publish_metrics."
                    ),
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                    },
                )
            if required_metric_names and not self.generation_evidence.has_required_queryability_dry_runs(
                required_metric_names
            ):
                missing_contracts = self.generation_evidence.missing_queryability_contracts(required_metric_names)
                contract_summary = summarize_queryability_contracts(missing_contracts)
                return FuncToolResult(
                    success=0,
                    error=(
                        "query_metrics(dry_run=True) must pass with the source SQL group-by dimensions before "
                        "publishing metrics. Run a dry-run query for the generated metric names with the matching "
                        f"dimensions/time grain, fix semantic model join or dimension issues, and retry. Missing: {contract_summary}"
                    ),
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                        "queryability_contracts": missing_contracts,
                    },
                )

            # Auto-sync to Knowledge Base
            sync_result = self._sync_metric_to_db(
                abs_metric,
                abs_semantic_files,
                metric_sqls,
                metric_names_to_sync=metric_names_to_sync,
            )

            if not sync_result.get("success"):
                return FuncToolResult(
                    success=0,
                    error=f"Metric file written but KB sync failed: {sync_result.get('error', 'unknown')}",
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files,
                        "metric_sqls": metric_sqls,
                        "sync": sync_result,
                    },
                )

            self.generation_evidence.mark_kb_sync("metric", scoped_metric_names)
            if sync_result.get("semantic_synced"):
                self.generation_evidence.mark_kb_sync("semantic")
            self._semantic_object_exists_cache.clear()
            self._semantic_table_object_index = None

            return FuncToolResult(
                result={
                    "message": "Metric generation completed and synced to Knowledge Base",
                    "metric_file": metric_file,
                    "semantic_model_files": semantic_model_files,
                    "metric_sqls": metric_sqls,
                    "metric_output_bindings": metric_output_bindings,
                    "sync": sync_result,
                }
            )

        except Exception as e:
            logger.error(f"Error completing metric generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

    def _validate_metric_output_bindings(
        self,
        raw_bindings: Optional[Iterable[Dict[str, str]]],
        published_metric_names: Iterable[str],
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """Require a one-to-one binding for every planned final SELECT output."""
        expected = list(self.generation_evidence.required_metric_output_ids)
        if not expected:
            return None, []
        if not raw_bindings:
            return (
                "metric_output_bindings is required for SQL-backed metric generation. "
                "Bind every planned output_id to one published metric name.",
                [],
            )

        bindings: List[Dict[str, str]] = []
        malformed_indexes: List[int] = []
        for index, item in enumerate(raw_bindings):
            if not isinstance(item, dict):
                malformed_indexes.append(index)
                continue
            output_id = str(item.get("output_id") or "").strip()
            metric_name = str(item.get("metric_name") or "").strip()
            if not output_id or not metric_name:
                malformed_indexes.append(index)
                continue
            bindings.append({"output_id": output_id, "metric_name": metric_name})
        if malformed_indexes:
            indexes = ", ".join(str(index) for index in malformed_indexes)
            return (
                "Each metric output binding must contain non-empty output_id and metric_name strings. "
                f"Invalid indexes: {indexes}.",
                bindings,
            )

        output_counts: Dict[str, int] = {}
        metric_counts: Dict[str, int] = {}
        for binding in bindings:
            output_id = binding["output_id"]
            output_counts[output_id] = output_counts.get(output_id, 0) + 1
            normalized_metric = normalize_metric_name(binding["metric_name"])
            metric_counts[normalized_metric] = metric_counts.get(normalized_metric, 0) + 1

        expected_set = set(expected)
        missing = [output_id for output_id in expected if output_id not in output_counts]
        unknown = sorted(output_id for output_id in output_counts if output_id not in expected_set)
        duplicate_outputs = sorted(output_id for output_id, count in output_counts.items() if count > 1)
        duplicate_metrics = sorted(
            binding["metric_name"]
            for binding in bindings
            if metric_counts.get(normalize_metric_name(binding["metric_name"]), 0) > 1
        )
        published = {
            normalize_metric_name(name)
            for name in published_metric_names
            if isinstance(name, str) and normalize_metric_name(name)
        }
        unpublished_metrics = sorted(
            {
                binding["metric_name"]
                for binding in bindings
                if normalize_metric_name(binding["metric_name"]) not in published
            }
        )
        errors: List[str] = []
        if missing:
            errors.append(f"missing output_ids: {', '.join(missing)}")
        if unknown:
            errors.append(f"unknown output_ids: {', '.join(unknown)}")
        if duplicate_outputs:
            errors.append(f"duplicate output_ids: {', '.join(duplicate_outputs)}")
        if duplicate_metrics:
            errors.append(f"metrics bound to multiple outputs: {', '.join(sorted(set(duplicate_metrics)))}")
        if unpublished_metrics:
            errors.append(f"metric names not published from metric_file: {', '.join(unpublished_metrics)}")
        if errors:
            return "Metric output coverage validation failed: " + "; ".join(errors) + ".", bindings
        return None, bindings

    @staticmethod
    def _validate_metric_file_has_blocks(metric_file: str) -> Optional[str]:
        """Return an actionable error string when ``metric_file`` lacks any
        named ``metric:`` YAML document; return ``None`` when at least one is found.

        The check is intentionally narrow: it verifies the file parses as YAML
        and at least one document carries a top-level ``metric:`` mapping with
        a non-empty ``name``. The downstream sync path still owns full metric
        schema handling.
        """
        if not metric_file or not os.path.exists(metric_file):
            return f"Metric file not found: {metric_file!r}"
        try:
            with open(metric_file, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
        except yaml.YAMLError as e:
            return (
                f"Metric file {metric_file!r} is not valid YAML ({e}). "
                "Rewrite the file with explicit `metric:` YAML blocks "
                "(separated by `---`)."
            )
        saw_metric_block = False
        seen_names: Dict[str, str] = {}
        saw_named_metric = False
        for doc in docs:
            if isinstance(doc, dict) and "metric" in doc:
                saw_metric_block = True
                metric = doc.get("metric")
                if isinstance(metric, dict):
                    name = metric.get("name")
                    if isinstance(name, str) and name.strip():
                        saw_named_metric = True
                        metric_name = name.strip()
                        normalized = normalize_metric_name(metric_name)
                        if normalized in seen_names:
                            return (
                                f"Metric file {metric_file!r} declares duplicate metric.name '{metric_name}'. "
                                "Metric names must be unique within a datasource; merge identical definitions "
                                "or choose a more specific business name."
                            )
                        seen_names[normalized] = metric_name
        if saw_named_metric:
            return None
        if saw_metric_block:
            return (
                f"Metric file {metric_file!r} contains `metric:` YAML blocks, "
                "but none has a non-empty `metric.name`. Rewrite the file with "
                "one explicit named metric per YAML document, for example: "
                "`metric: {name: revenue_total, type: measure_proxy, type_params: {measure: revenue_total}}`."
            )
        return (
            f"Metric file {metric_file!r} contains no `metric:` YAML blocks. "
            "Documentation/markdown is not a metric definition. Rewrite the file "
            "with explicit `metric:` entries (separated by `---`); do not rely on "
            "`create_metric: true` on semantic-model measures — those only emit "
            "metrics at MetricFlow runtime and are NOT synced to the Knowledge Base."
        )

    @staticmethod
    def _extract_metric_names_from_file(metric_file: str) -> List[str]:
        """Return metric names declared in top-level ``metric:`` YAML blocks."""
        try:
            with open(metric_file, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
        except (OSError, yaml.YAMLError):
            return []

        names: List[str] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            metric = doc.get("metric")
            if not isinstance(metric, dict):
                continue
            name = metric.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return names

    @staticmethod
    def _extract_metric_definitions_from_file(metric_file: str) -> List[Dict[str, object]]:
        """Return lightweight metric definitions for pre-sync conflict checks."""
        try:
            with open(metric_file, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
        except (OSError, yaml.YAMLError):
            return []

        definitions: List[Dict[str, object]] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            metric = doc.get("metric")
            if not isinstance(metric, dict):
                continue
            name = metric.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            metric_type = str(metric.get("type") or "").strip()
            type_params = metric.get("type_params") if isinstance(metric.get("type_params"), dict) else {}
            measure_expr = ""
            base_measures: List[str] = []

            if metric_type == "measure_proxy":
                measure = type_params.get("measure")
                if isinstance(measure, str):
                    measure_expr = measure
                    base_measures.append(measure)
                elif isinstance(measure, dict):
                    measure_name = measure.get("name")
                    if isinstance(measure_name, str) and measure_name.strip():
                        measure_expr = measure_name
                        base_measures.append(measure_name)
            elif metric_type == "ratio":
                for key in ("numerator", "denominator"):
                    ref = type_params.get(key)
                    if isinstance(ref, str) and ref.strip():
                        base_measures.append(ref)
                    elif isinstance(ref, dict):
                        ref_name = ref.get("name")
                        if isinstance(ref_name, str) and ref_name.strip():
                            base_measures.append(ref_name)
            elif metric_type in {"expr", "cumulative"}:
                for ref in type_params.get("measures") or []:
                    if isinstance(ref, str) and ref.strip():
                        base_measures.append(ref)
                    elif isinstance(ref, dict):
                        ref_name = ref.get("name")
                        if isinstance(ref_name, str) and ref_name.strip():
                            base_measures.append(ref_name)
                if metric_type == "expr" and type_params.get("expr"):
                    measure_expr = str(type_params["expr"])
            elif metric_type == "derived":
                for ref in type_params.get("metrics") or []:
                    if isinstance(ref, str) and ref.strip():
                        base_measures.append(ref)
                    elif isinstance(ref, dict):
                        ref_name = ref.get("name")
                        if isinstance(ref_name, str) and ref_name.strip():
                            base_measures.append(ref_name)
                if type_params.get("expr"):
                    measure_expr = str(type_params["expr"])

            definitions.append(
                {
                    "name": name.strip(),
                    "metric_type": metric_type,
                    "measure_expr": measure_expr,
                    "base_measures": base_measures,
                }
            )
        return definitions

    def _metric_names_requiring_dry_run(
        self,
        metric_names: List[str],
        metric_definitions: List[Dict[str, object]],
        metric_sqls: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Return the subset of a metric file that must be dry-run before publish.

        Batch generation appends new metrics to existing metrics YAML files. Later
        batches should not have to re-dry-run every historical metric already in
        that file; they only need to dry-run new metrics and metrics with SQL
        evidence produced in this node run.
        """
        if not metric_names:
            return []

        by_normalized_name = {normalize_metric_name(name): name for name in metric_names if normalize_metric_name(name)}
        required = set()
        for name in metric_sqls or {}:
            normalized = normalize_metric_name(name)
            if normalized and not normalized.startswith("__") and normalized in by_normalized_name:
                required.add(normalized)
        for name in self.generation_evidence.metric_dry_run_metrics:
            normalized = normalize_metric_name(name)
            if normalized in by_normalized_name:
                required.add(normalized)

        existing_names = self._existing_metric_names()
        if existing_names is None:
            return metric_names
        for definition in metric_definitions:
            normalized = normalize_metric_name(definition.get("name"))
            if normalized and normalized not in existing_names:
                required.add(normalized)

        if not required:
            return []
        return [name for normalized, name in by_normalized_name.items() if normalized in required]

    @staticmethod
    def _filter_metric_names(metric_names: List[str], metric_names_to_sync: Optional[set[str]]) -> List[str]:
        if metric_names_to_sync is None:
            return metric_names
        return [name for name in metric_names if normalize_metric_name(name) in metric_names_to_sync]

    @staticmethod
    def _filter_metric_definitions(
        metric_definitions: List[Dict[str, object]], metric_names_to_sync: Optional[set[str]]
    ) -> List[Dict[str, object]]:
        if metric_names_to_sync is None:
            return metric_definitions
        return [
            definition
            for definition in metric_definitions
            if normalize_metric_name(definition.get("name")) in metric_names_to_sync
        ]

    @staticmethod
    def _public_metric_sql_names(metric_sqls: Optional[Dict[str, str]]) -> set[str]:
        names: set[str] = set()
        for name in metric_sqls or {}:
            raw_name = str(name or "").strip()
            if not raw_name or raw_name.startswith("__"):
                continue
            normalized = normalize_metric_name(raw_name)
            if normalized:
                names.add(normalized)
        return names

    def _metric_names_to_sync(
        self, metric_names: List[str], metric_sqls: Optional[Dict[str, str]]
    ) -> Optional[set[str]]:
        """Return the metric subset to publish, or None to publish the full file.

        Later bootstrap batches often append one new metric to a file that already
        contains older metrics. Treat request-local dry-run SQL keys and metric
        names as the current publish scope so historical metrics in the same YAML
        file are not re-upserted.
        """

        dry_run_names = {
            normalized
            for normalized in (normalize_metric_name(name) for name in self.generation_evidence.metric_dry_run_metrics)
            if normalized
        }
        scope_names = self._public_metric_sql_names(metric_sqls) | dry_run_names
        if not scope_names:
            return None
        if not metric_names:
            return None

        declared_names = {normalize_metric_name(name) for name in metric_names if normalize_metric_name(name)}
        matched_names = scope_names & declared_names
        if not matched_names:
            return set()

        existing_names = self._existing_metric_names()
        if existing_names is not None:
            new_names = matched_names - existing_names
            if new_names:
                return new_names
        return matched_names

    def _existing_metric_names(self) -> Optional[set[str]]:
        try:
            rows = self.metric_rag.search_all_metrics(select_fields=["name"])
        except Exception as exc:
            logger.warning("Failed to load existing metric names before publish dry-run gating: %s", exc)
            return None
        if not _is_supported_row_container(rows):
            return None
        names = set()
        for row in _rows_to_dicts(rows):
            normalized = normalize_metric_name(row.get("name"))
            if normalized:
                names.add(normalized)
        return names

    def _validate_metric_name_conflicts(self, metric_definitions: List[Dict[str, object]]) -> Optional[str]:
        if not metric_definitions:
            return None

        existing_by_name: Dict[str, List[Dict[str, object]]] = {}
        try:
            rows = self.metric_rag.search_all_metrics(
                select_fields=["id", "name", "semantic_model_name", "metric_type", "measure_expr", "base_measures"]
            )
        except Exception as exc:
            logger.warning("Failed to check existing metric name conflicts before sync: %s", exc)
            return None
        if not _is_supported_row_container(rows):
            return None

        for row in _rows_to_dicts(rows):
            normalized = normalize_metric_name(row.get("name"))
            if normalized:
                existing_by_name.setdefault(normalized, []).append(row)

        for incoming in metric_definitions:
            normalized = normalize_metric_name(incoming.get("name"))
            if not normalized:
                continue
            for existing in existing_by_name.get(normalized, []):
                conflict_field = metric_definition_conflict(existing, incoming)
                if conflict_field:
                    return (
                        f"Metric name conflict within this datasource for '{incoming.get('name')}': "
                        f"existing metric id '{existing.get('id')}' has a different '{conflict_field}'. "
                        "Metric names must be unique within a datasource; choose a more specific name "
                        "or update the existing metric explicitly."
                    )
        return None

    def _resolve_generation_path(self, path: str, kind: str) -> str:
        if not path:
            return ""
        from datus.cli.generation_hooks import resolve_kb_sandbox_path

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

    def _validate_required_query_backed_sources(
        self,
        extra_paths: Optional[List[str]] = None,
        *,
        include_active_root: bool = True,
    ) -> Optional[str]:
        """Require every planned query-backed SQL to exist in the active YAML tree."""
        required = self.generation_evidence.required_query_backed_sql
        if not required:
            return None

        path_manager = get_path_manager(agent_config=self.agent_config)
        datasource = getattr(self.agent_config, "current_datasource", "")
        semantic_model_path = getattr(path_manager, "semantic_model_path", None)
        if isinstance(datasource, str) and datasource.strip() and callable(semantic_model_path):
            active_semantic_root = Path(semantic_model_path(datasource))
        else:
            active_semantic_root = Path(path_manager.subject_dir) / "semantic_models"

        search_paths = [active_semantic_root] if include_active_root else []
        active_root_resolved = active_semantic_root.expanduser().resolve(strict=False)
        for path in extra_paths or []:
            if not path:
                continue
            candidate = Path(path).expanduser().resolve(strict=False)
            try:
                candidate.relative_to(active_root_resolved)
            except ValueError:
                continue
            search_paths.append(candidate)
        seen_paths: set[str] = set()
        authored_sources: set[str] = set()
        for search_path in search_paths:
            normalized_path = str(search_path.expanduser().resolve(strict=False))
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            for document in self._iter_yaml_docs(normalized_path):
                data_sources = document.get("data_source")
                if isinstance(data_sources, dict):
                    data_sources = [data_sources]
                for data_source in data_sources or []:
                    if not isinstance(data_source, dict):
                        continue
                    sql_query = data_source.get("sql_query")
                    if isinstance(sql_query, str) and sql_query.strip():
                        authored_sources.add(self._normalize_query_source(sql_query))

                semantic_models = document.get("semantic_model")
                for semantic_model in semantic_models if isinstance(semantic_models, list) else []:
                    if not isinstance(semantic_model, dict):
                        continue
                    for dataset in semantic_model.get("datasets") or []:
                        if not isinstance(dataset, dict):
                            continue
                        source = dataset.get("source")
                        if isinstance(source, str) and source.strip():
                            authored_sources.add(self._normalize_query_source(source))

        missing = [
            requirement_id
            for requirement_id, sql in required.items()
            if self._normalize_query_source(sql) not in authored_sources
        ]
        if not missing:
            return None
        return (
            "Every query-backed dataset must preserve the exact SQL returned by prepare_sql_modeling_plan. "
            "Missing or rewritten dataset requirements: "
            f"{', '.join(missing)}."
        )

    @staticmethod
    def _normalize_query_source(value: str) -> str:
        """Normalize transport-only whitespace while preserving SQL content."""
        return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    def _validate_required_osi_group_by_dimensions(
        self,
        semantic_model_file: str | Path,
    ) -> tuple[str, List[Dict[str, str]]]:
        """Require executable OSI dimensions for every non-time GROUP BY expression."""
        try:
            db_config = self.agent_config.current_db_config()
        except Exception:
            db_config = None
        configured_type = getattr(db_config, "type", "") if db_config is not None else ""
        dialect = configured_type if isinstance(configured_type, str) else ""

        required: List[Dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for contract in self.generation_evidence.metric_queryability_contracts:
            for hint in contract.get("dimension_expr_hints") or []:
                if not isinstance(hint, dict):
                    continue
                normalized = {
                    "source": str(contract.get("source") or "source SQL").strip(),
                    "alias": str(hint.get("alias") or "").strip(),
                    "expr": str(hint.get("expr") or "").strip(),
                    "column": str(hint.get("column") or "").strip(),
                }
                identity = (
                    _normalized_semantic_name(normalized["alias"]),
                    _canonical_sql_expression(normalized["expr"], dialect),
                    _normalized_semantic_name(normalized["column"]),
                )
                if not identity[0] or not identity[1] or identity in seen:
                    continue
                seen.add(identity)
                required.append(normalized)
        if not required:
            return "", []

        dimensions = _osi_dimension_fields(semantic_model_file)
        missing: List[Dict[str, str]] = []
        for hint in required:
            required_expression = _canonical_sql_expression(hint["expr"], dialect)
            required_column = _canonical_sql_expression(hint["column"], dialect)
            is_direct_column = bool(required_column and required_expression == required_column)
            matched = False
            for field in dimensions:
                field_name = _normalized_semantic_name(field.get("name"))
                field_expressions = {
                    canonical
                    for expression in field.get("expressions") or []
                    if (canonical := _canonical_sql_expression(expression, dialect))
                }
                if is_direct_column:
                    accepted_names = {
                        _normalized_semantic_name(hint["alias"]),
                        _normalized_semantic_name(hint["column"]),
                    }
                    matched = field_name in accepted_names and required_expression in field_expressions
                else:
                    matched = (
                        field_name == _normalized_semantic_name(hint["alias"])
                        and required_expression in field_expressions
                    )
                if matched:
                    break
            if not matched:
                missing.append(hint)

        if not missing:
            return "", []

        fixes = "; ".join(
            f"{hint['source']}: add dimension field `{hint['alias']}` with expression `{hint['expr']}`"
            for hint in missing
        )
        return (
            "The OSI semantic model does not expose every non-time source SQL GROUP BY expression "
            f"as an executable dimension. {fixes}. A raw dependency column or description text does "
            "not satisfy a derived expression. Update the dataset, run validate_semantic again, then "
            "retry publish_semantic_model.",
            missing,
        )

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

    def _osi_document_root(self, metric_file: Optional[str] = None, semantic_model_file: Optional[str] = None) -> str:
        """Resolve the datasource-scoped OSI directory used by the adapter."""
        candidates: List[Path] = []
        try:
            datasource = getattr(self.agent_config, "current_datasource", "")
            path_manager = getattr(self.agent_config, "path_manager", None)
            if datasource and path_manager and hasattr(path_manager, "semantic_model_path"):
                candidates.append(Path(path_manager.semantic_model_path(datasource)))
        except Exception:
            pass

        for raw in (semantic_model_file, metric_file):
            if not raw:
                continue
            path = Path(raw)
            if path.is_file():
                parent = path.parent
                candidates.append(parent.parent if parent.name == "metrics" else parent)
            elif path.is_dir():
                candidates.append(path)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0]) if candidates else str(Path(metric_file or semantic_model_file or ".").parent)

    def _load_osi_document(
        self,
        metric_file: Optional[str] = None,
        semantic_model_file: Optional[str] = None,
    ):
        from datus_semantic_osi.profile import load_osi_model

        model_names = self.extract_osi_model_names(metric_file) if metric_file else []
        if not model_names and semantic_model_file:
            model_names = self.extract_osi_model_names(semantic_model_file)
        if len(model_names) != 1:
            candidates = ", ".join(model_names) or "<none>"
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message=(
                    f"Exactly one semantic model must be identifiable from the target artifact. Found: {candidates}."
                ),
            )
        return load_osi_model(
            self._osi_document_root(metric_file, semantic_model_file),
            semantic_model_name=model_names[0],
            normalize=True,
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

        for dim in getattr(dataset, "dimensions", []):
            dim_name = getattr(dim, "name", "")
            if not dim_name:
                continue
            columns.append(
                {
                    "name": str(dim_name),
                    "expr": getattr(dim, "expr", None) or str(dim_name),
                    "role": "dimension",
                    "type": str(getattr(dim, "type", "") or ""),
                    "granularity": getattr(dim, "granularity", "") or "",
                    "description": getattr(dim, "description", "") or "",
                    "ai_context": getattr(dim, "ai_context", None),
                }
            )
        return [{key: value for key, value in item.items() if value not in (None, "", [], {})} for item in columns]

    @classmethod
    def _osi_dataset_relationships(cls, doc: Any, dataset_name: str) -> List[dict]:
        relationships: List[dict] = []
        for relationship in getattr(doc, "relationships", []) or []:
            from_dataset = cls._relationship_endpoint(relationship, "from", "from_dataset")
            to_dataset = cls._relationship_endpoint(relationship, "to", "to_dataset")
            if dataset_name not in (from_dataset, to_dataset):
                continue
            from_columns = cls._relationship_columns(relationship, "from_columns", "from_identifier")
            to_columns = cls._relationship_columns(relationship, "to_columns", "to_identifier")
            relationships.append(
                {
                    "name": str(getattr(relationship, "name", "") or ""),
                    "type": str(getattr(relationship, "type", "") or ""),
                    "from_dataset": from_dataset,
                    "to_dataset": to_dataset,
                    "from_columns": from_columns,
                    "to_columns": to_columns,
                    "role": "from" if from_dataset == dataset_name else "to",
                    "ai_context": getattr(relationship, "ai_context", None),
                }
            )
        return [
            {key: value for key, value in item.items() if value not in (None, "", [], {})} for item in relationships
        ]

    @classmethod
    def _osi_table_semantic_profile(
        cls,
        *,
        doc: Any,
        dataset: Any,
        table_name: str,
        table_fq_name: str,
        db_parts: dict[str, str],
        yaml_path: str,
    ) -> dict:
        dataset_name = str(getattr(dataset, "name", "") or table_name)
        columns = cls._osi_dataset_columns(dataset)
        relationships = cls._osi_dataset_relationships(doc, dataset_name)
        ai_context = getattr(dataset, "ai_context", None)
        custom_extensions = getattr(dataset, "custom_extensions", None) or []
        description = getattr(dataset, "description", "") or ""
        semantic_model_name = str(getattr(doc, "name", "") or "")
        physical_table = table_fq_name or table_name
        return {
            "id": f"osi:{semantic_model_name}:{physical_table}",
            "format": "osi",
            "physical_table_fq_name": physical_table,
            "table_name": table_name,
            "semantic_model_name": semantic_model_name,
            "dataset_name": dataset_name,
            "data_source_name": "",
            "description": description,
            "ai_context_json": cls._json_dumps(ai_context),
            "columns_json": cls._json_dumps(columns),
            "relationships_json": cls._json_dumps(relationships),
            "custom_extensions_json": cls._json_dumps(custom_extensions),
            "yaml_path": yaml_path,
            "search_text": cls._profile_search_text(
                semantic_model_name,
                dataset_name,
                physical_table,
                description,
                ai_context,
                columns,
                relationships,
            ),
            "updated_at": datetime.now().replace(microsecond=0),
            **db_parts,
        }

    def _upsert_table_semantic_profiles(self, profiles: List[dict]) -> int:
        if not profiles or self.table_semantic_profile_rag is None:
            return 0
        yaml_path = str(profiles[0].get("yaml_path") or "")
        self.table_semantic_profile_rag.upsert_batch(profiles)
        self.table_semantic_profile_rag.create_indices()
        if yaml_path:
            self.table_semantic_profile_rag.delete_artifact_rows_except(
                yaml_path, [profile.get("id", "") for profile in profiles]
            )
        return len(profiles)

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

    @staticmethod
    def _dataset_table_name(dataset: Any) -> str:
        source = getattr(dataset, "source", None)
        table = getattr(source, "table", None) or getattr(dataset, "name", "")
        return str(table).split(".")[-1]

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

    def _sync_osi_semantic_objects_to_db(self, semantic_model_path: str) -> dict:
        """Sync OSI datasets into the semantic object store."""
        try:
            target_dataset_names = set(self.extract_osi_dataset_names(semantic_model_path))
            if not target_dataset_names:
                return {
                    "success": False,
                    "error": f"No OSI datasets found in semantic model file to sync: {semantic_model_path}",
                }

            doc = self._load_osi_document(semantic_model_file=semantic_model_path)
            semantic_model_name = str(getattr(doc, "name", "") or "")
            default_db_parts = self._current_db_parts(self.agent_config)
            semantic_objects: List[dict] = []
            table_profiles: List[dict] = []
            synced_items: List[str] = []

            for dataset in getattr(doc, "datasets", []):
                dataset_name = getattr(dataset, "name", "")
                if dataset_name not in target_dataset_names:
                    continue
                table_name = self._dataset_table_name(dataset)
                db_parts = self._dataset_db_parts(dataset, default_db_parts)
                fq_parts = [db_parts["catalog_name"], db_parts["database_name"], db_parts["schema_name"], table_name]
                table_fq_name = ".".join(part for part in fq_parts if part)
                yaml_path = semantic_model_path
                table_profiles.append(
                    self._osi_table_semantic_profile(
                        doc=doc,
                        dataset=dataset,
                        table_name=table_name,
                        table_fq_name=table_fq_name,
                        db_parts=db_parts,
                        yaml_path=yaml_path,
                    )
                )

                semantic_objects.append(
                    {
                        "id": f"table:{semantic_model_name}:{table_fq_name}",
                        "kind": "table",
                        "name": table_name,
                        "fq_name": table_fq_name,
                        "table_name": table_name,
                        "description": getattr(dataset, "description", "") or "",
                        "yaml_path": yaml_path,
                        "updated_at": datetime.now().replace(microsecond=0),
                        **db_parts,
                        "semantic_model_name": semantic_model_name,
                        "is_dimension": False,
                        "is_measure": False,
                        "is_entity_key": False,
                        "is_deprecated": False,
                        "expr": "",
                        "column_type": "",
                        "agg": "",
                        "create_metric": False,
                        "agg_time_dimension": "",
                        "is_partition": False,
                        "time_granularity": "",
                        "entity": "",
                    }
                )
                synced_items.append(f"table:{table_fq_name}")

                primary_keys = getattr(dataset, "primary_key", None) or []
                if isinstance(primary_keys, str):
                    primary_keys = [primary_keys]
                for key in primary_keys:
                    semantic_objects.append(
                        self._osi_column_object(
                            table_name=table_name,
                            table_fq_name=table_fq_name,
                            semantic_model_name=semantic_model_name,
                            name=str(key),
                            description="Primary key",
                            expr=str(key),
                            column_type="PRIMARY",
                            yaml_path=yaml_path,
                            db_parts=db_parts,
                            is_entity_key=True,
                        )
                    )

                time_dimension = getattr(dataset, "time_dimension", None)
                if time_dimension and getattr(time_dimension, "name", None):
                    semantic_objects.append(
                        self._osi_column_object(
                            table_name=table_name,
                            table_fq_name=table_fq_name,
                            semantic_model_name=semantic_model_name,
                            name=str(time_dimension.name),
                            description="Primary time dimension",
                            expr=str(time_dimension.name),
                            column_type="TIME",
                            yaml_path=yaml_path,
                            db_parts=db_parts,
                            is_dimension=True,
                            time_granularity=getattr(time_dimension, "granularity", "") or "",
                        )
                    )

                for dim in getattr(dataset, "dimensions", []):
                    dim_name = getattr(dim, "name", "")
                    if not dim_name:
                        continue
                    semantic_objects.append(
                        self._osi_column_object(
                            table_name=table_name,
                            table_fq_name=table_fq_name,
                            semantic_model_name=semantic_model_name,
                            name=str(dim_name),
                            description=getattr(dim, "description", "") or "",
                            expr=getattr(dim, "expr", None) or str(dim_name),
                            column_type=str(getattr(dim, "type", "") or ""),
                            yaml_path=yaml_path,
                            db_parts=db_parts,
                            is_dimension=True,
                            time_granularity=getattr(dim, "granularity", "") or "",
                        )
                    )

            if not semantic_objects:
                return {
                    "success": False,
                    "error": (
                        "OSI datasets declared in semantic model file were not found after loading datasource "
                        f"context: {', '.join(sorted(target_dataset_names))}"
                    ),
                }
            replacement_plans = [(self.semantic_rag, semantic_model_path, semantic_objects)]
            if table_profiles and self.table_semantic_profile_rag is not None:
                replacement_plans.append((self.table_semantic_profile_rag, semantic_model_path, table_profiles))
            snapshots = snapshot_artifact_replacements(replacement_plans)
            try:
                self.semantic_rag.upsert_batch(semantic_objects)
                self.semantic_rag.create_indices()
                profile_count = 0
                if table_profiles and self.table_semantic_profile_rag is not None:
                    self.table_semantic_profile_rag.upsert_batch(table_profiles)
                    self.table_semantic_profile_rag.create_indices()
                    profile_count = len(table_profiles)
                delete_stale_artifact_rows(replacement_plans)
            except Exception as sync_exc:
                restore_failures = restore_artifact_replacements(snapshots)
                if restore_failures:
                    raise RuntimeError(
                        "OSI semantic replacement failed and rollback was incomplete for: "
                        f"{', '.join(restore_failures)}"
                    ) from sync_exc
                raise
            # Post-commit, best-effort cleanup of shadowed stale rows; never raises.
            self.semantic_rag.delete_shadowed_table_rows(semantic_objects)
            return {
                "success": True,
                "message": f"Synced {len(semantic_objects)} OSI semantic object(s): {', '.join(synced_items[:5])}",
                "semantic_objects": len(semantic_objects),
                "table_semantic_profiles": profile_count,
            }
        except Exception as e:
            logger.error(f"Error syncing OSI semantic objects to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _osi_column_object(
        *,
        table_name: str,
        table_fq_name: str,
        semantic_model_name: str,
        name: str,
        description: str,
        expr: str,
        column_type: str,
        yaml_path: str,
        db_parts: dict[str, str],
        is_dimension: bool = False,
        is_entity_key: bool = False,
        time_granularity: str = "",
    ) -> dict:
        return {
            "id": f"column:{semantic_model_name}:{table_fq_name}.{name}",
            "kind": "column",
            "name": name,
            "fq_name": f"{table_fq_name}.{name}",
            "table_name": table_name,
            "description": description,
            "is_dimension": is_dimension,
            "is_measure": False,
            "is_entity_key": is_entity_key,
            "is_deprecated": False,
            "yaml_path": yaml_path,
            "updated_at": datetime.now().replace(microsecond=0),
            **db_parts,
            "semantic_model_name": semantic_model_name,
            "expr": expr,
            "column_type": column_type,
            "agg": "",
            "create_metric": False,
            "agg_time_dimension": "",
            "is_partition": False,
            "time_granularity": time_granularity,
            "entity": name if is_entity_key else "",
        }

    def _sync_osi_metric_to_db(
        self,
        metric_file: str,
        semantic_model_file: Optional[str | List[str]] = None,
        metric_sqls: Optional[Dict[str, str]] = None,
        metric_names_to_sync: Optional[set[str]] = None,
    ) -> dict:
        """Sync all OSI metrics, or incrementally upsert one explicit metric scope."""
        try:
            semantic_model_files = (
                list(semantic_model_file)
                if isinstance(semantic_model_file, list)
                else ([semantic_model_file] if semantic_model_file else [])
            )
            declared_metric_names = set(self.extract_osi_metric_names(metric_file))
            if not declared_metric_names:
                return {"success": False, "error": f"No OSI metrics found in metric file to sync: {metric_file}"}
            target_metric_names = (
                declared_metric_names
                if metric_names_to_sync is None
                else {str(name).strip() for name in metric_names_to_sync if str(name).strip()}
            )
            missing_metric_names = target_metric_names - declared_metric_names
            if missing_metric_names:
                return {
                    "success": False,
                    "error": (
                        "OSI metric publish scope contains names not declared in the metric file: "
                        f"{', '.join(sorted(missing_metric_names))}"
                    ),
                }
            if not target_metric_names:
                return {"success": False, "error": "OSI metric publish scope must not be empty"}

            doc = self._load_osi_document(
                metric_file=metric_file,
                semantic_model_file=semantic_model_files[0] if semantic_model_files else None,
            )
            semantic_model_name = str(getattr(doc, "name", "") or "")
            db_parts = self._current_db_parts(self.agent_config)
            metric_objects: List[dict] = []
            synced_items: List[str] = []

            for metric in getattr(doc, "metrics", []):
                metric_name = getattr(metric, "name", "")
                if not metric_name:
                    continue
                if metric_name not in target_metric_names:
                    continue
                dimensions = self._metric_query_dimensions(doc, metric)
                entities = self._metric_entities(doc, metric)

                subject_path = self._metric_subject_path(metric)
                measure_expr = self._metric_expression(metric)
                metric_obj = {
                    "name": metric_name,
                    "subject_path": subject_path,
                    "semantic_model_name": semantic_model_name,
                    "id": build_metric_id(subject_path, metric_name),
                    "description": getattr(metric, "description", "") or "",
                    "metric_type": getattr(metric, "kind", None) or "aggregate",
                    "measure_expr": measure_expr,
                    "base_measures": [measure_expr] if measure_expr else [],
                    "dimensions": dimensions,
                    "entities": entities,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": datetime.now().replace(microsecond=0),
                    **db_parts,
                    "sql": metric_sqls.get(metric_name, "") if metric_sqls else "",
                    "yaml_path": metric_file,
                }
                metric_objects.append(metric_obj)
                synced_items.append(f"metric:{metric_name}")

            if not metric_objects:
                return {
                    "success": False,
                    "error": (
                        "OSI metrics declared in metric file were not found after loading datasource context: "
                        f"{', '.join(sorted(target_metric_names))}"
                    ),
                }

            synced_semantic_files: List[str] = []
            for current_semantic_file in semantic_model_files:
                sem_result = self._sync_osi_semantic_objects_to_db(current_semantic_file)
                if not sem_result.get("success"):
                    return sem_result
                synced_semantic_files.append(current_semantic_file)

            metric_plan = (self.metric_rag, metric_file, metric_objects)
            replacement_plans = [metric_plan] if metric_names_to_sync is None else []
            snapshots = snapshot_artifact_replacements([metric_plan])
            try:
                self.metric_rag.upsert_batch(metric_objects)
                self.metric_rag.create_indices()
                delete_stale_artifact_rows(replacement_plans)
            except Exception as sync_exc:
                restore_failures = restore_artifact_replacements(snapshots)
                if restore_failures:
                    raise RuntimeError(
                        f"OSI metric replacement failed and rollback was incomplete for: {', '.join(restore_failures)}"
                    ) from sync_exc
                raise
            return {
                "success": True,
                "message": f"Synced {len(metric_objects)} OSI metric(s): {', '.join(synced_items[:5])}",
                "metric_artifact_ids": [obj["id"] for obj in metric_objects],
                "metric_names": [obj["name"] for obj in metric_objects],
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

        This is the low-level import/refresh entry used outside request-local
        generation. Agent authoring must publish through ``publish_semantic_model``
        or ``publish_metrics`` so validation and dry-run gates remain enforced.

        A metrics-bearing document is synced in one pass: ``_sync_osi_metric_to_db``
        already vectorizes the referenced datasets before the metrics when a
        ``semantic_model_file`` is given, so no separate dataset sync is needed. A
        dataset-only document syncs just its datasets. The result always carries a
        ``synced`` count for uniform accounting across both shapes.
        """
        try:
            if not include_semantic_objects and not include_metrics:
                return {"success": False, "error": "At least one OSI sync scope must be enabled", "synced": 0}
            if include_metrics and self.extract_osi_metric_names(osi_file_path):
                result = self._sync_osi_metric_to_db(
                    metric_file=osi_file_path,
                    semantic_model_file=osi_file_path if include_semantic_objects else None,
                )
                synced = len(result.get("metric_artifact_ids") or [])
            elif include_metrics and not include_semantic_objects:
                result = self._sync_osi_metric_to_db(metric_file=osi_file_path)
                synced = len(result.get("metric_artifact_ids") or [])
            else:
                result = self._sync_osi_semantic_objects_to_db(osi_file_path)
                synced = int(result.get("semantic_objects") or 0)
            return {**result, "synced": synced}
        except Exception as e:
            logger.error(f"Error syncing OSI document to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _sync_metric_to_db(
        self,
        metric_file: str,
        semantic_model_files: Optional[List[str]] = None,
        metric_sqls: Optional[Dict[str, str]] = None,
        metric_names_to_sync: Optional[set[str]] = None,
    ) -> dict:
        """
        Sync metric and any updated semantic models to Knowledge Base.

        Reuses GenerationHooks._sync_semantic_to_db() static method.

        Args:
            metric_file: Absolute path to metric YAML file
            semantic_model_files: Optional absolute paths to semantic model YAML files
            metric_sqls: Optional dict mapping metric names to generated SQL

        Returns:
            dict with sync result (success, message, or error)
        """
        from datus.cli.generation_hooks import GenerationHooks

        try:
            if not os.path.exists(metric_file):
                return {"success": False, "error": f"Metric file not found: {metric_file}"}

            synced_semantic_files: List[str] = []
            for semantic_model_file in semantic_model_files or []:
                if not os.path.exists(semantic_model_file):
                    return {
                        "success": False,
                        "error": f"Semantic model file not found: {semantic_model_file}",
                    }
                sem_result = GenerationHooks._sync_semantic_to_db(
                    semantic_model_file,
                    self.agent_config,
                    include_semantic_objects=True,
                    include_metrics=False,
                )
                if not sem_result.get("success"):
                    return sem_result
                synced_semantic_files.append(semantic_model_file)

            sync_metric_file = metric_file
            temp_metric_file = ""
            sync_metric_sqls = metric_sqls
            if metric_names_to_sync is not None:
                sync_metric_file = self._write_filtered_metric_file(metric_file, metric_names_to_sync)
                temp_metric_file = sync_metric_file if sync_metric_file != metric_file else ""
                sync_metric_sqls = self._filter_metric_sqls(metric_sqls, metric_names_to_sync)

            try:
                result = GenerationHooks._sync_semantic_to_db(
                    sync_metric_file,
                    self.agent_config,
                    include_semantic_objects=False,
                    include_metrics=True,
                    metric_sqls=sync_metric_sqls,
                    original_yaml_path=metric_file,
                    replace_metric_artifact=False,
                )
            finally:
                if temp_metric_file and os.path.exists(temp_metric_file):
                    os.remove(temp_metric_file)
            if result.get("success"):
                result["semantic_synced"] = bool(synced_semantic_files)
                result["semantic_model_files_synced"] = synced_semantic_files
                if metric_names_to_sync is not None:
                    result["metric_names_synced"] = sorted(metric_names_to_sync)

            if result.get("success"):
                logger.info(f"Successfully synced metric to KB: {result.get('message')}")
            else:
                logger.error(f"Failed to sync metric to KB: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Error syncing metric to KB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _filter_metric_sqls(
        metric_sqls: Optional[Dict[str, str]], metric_names_to_sync: set[str]
    ) -> Optional[Dict[str, str]]:
        if metric_sqls is None:
            return None
        filtered: Dict[str, str] = {}
        for name, sql in metric_sqls.items():
            normalized = normalize_metric_name(name)
            if normalized in metric_names_to_sync:
                filtered[name] = sql
        return filtered

    @staticmethod
    def _write_filtered_metric_file(metric_file: str, metric_names_to_sync: set[str]) -> str:
        with open(metric_file, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))

        filtered_docs = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            metric = doc.get("metric")
            if not isinstance(metric, dict):
                continue
            if normalize_metric_name(metric.get("name")) in metric_names_to_sync:
                filtered_docs.append(doc)

        if not filtered_docs:
            raise ValueError("No matching metric definitions found for current publish scope")

        fd, temp_path = tempfile.mkstemp(
            prefix=".metric_publish_",
            suffix=".yml",
            dir=os.path.dirname(metric_file),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump_all(filtered_docs, f, allow_unicode=True, sort_keys=False)
        return temp_path

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
