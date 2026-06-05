# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import os
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

import yaml
from agents import Tool
from datus_storage_base.conditions import And, eq

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricRAG, metric_definition_conflict, normalize_metric_name
from datus.storage.semantic_model.store import SemanticModelRAG, _identifier_variants, _normalized_identifier
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.metric_queryability import summarize_queryability_contracts
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

logger = get_logger(__name__)


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


class GenerationTools:
    """
    Tools for semantic model generation workflow.

    This class provides tools for checking existing semantic models and
    completing the generation process.
    """

    def __init__(self, agent_config: AgentConfig, generation_evidence: Optional[GenerationEvidence] = None):
        self.agent_config = agent_config
        self.generation_evidence = generation_evidence or GenerationEvidence()
        self.metric_rag = MetricRAG(agent_config)
        self.semantic_rag = SemanticModelRAG(agent_config)
        self._semantic_object_exists_cache: Dict[tuple[str, str, str], FuncToolResult] = {}
        self._semantic_table_object_index: Optional[Dict[str, Dict[str, object]]] = None

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
                self.end_semantic_model_generation,
                self.end_metric_generation,
            )
        ]

    def check_semantic_object_exists(
        self,
        object_name: str,
        kind: str = "table",  # table, column, metric
        table_context: str = "",
    ) -> FuncToolResult:
        """
        Check if a semantic object (table, column, metric) already exists in vector store.

        Use this tool to avoid duplicating work.

        Args:
            object_name: Name of the object (e.g. "orders", "orders.amount")
            kind: Type of object ("table", "column", "metric")
            table_context: If checking a column/metric, providing the table name helps narrow search.

        Returns:
            dict: Check results containing existence status and details.
        """
        try:
            normalized_kind = str(kind or "").strip().lower()
            cache_key = (
                normalized_kind,
                str(object_name or "").strip().lower(),
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
                where = eq("name", target_name)
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
        rows = storage.search_all(where=And([eq("kind", "table")]), select_fields=select_fields)
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

    def end_semantic_model_generation(self, semantic_model_files: List[str]) -> FuncToolResult:
        """
        Complete semantic model generation process.

        Call this tool when you have finished generating semantic model YAML files.
        In interactive runs, the generation hook syncs these files to the Knowledge Base
        directly after this tool returns.

        Args:
            semantic_model_files: List of generated semantic model YAML file paths.
                Relative file names within the sub-agent's semantic-model workspace
                are preferred (e.g. ``["orders.yml", "customers.yml"]``). Absolute
                paths are also accepted. The downstream hook resolves relative
                entries against the live agent_config datasource.

        Returns:
            dict: Result containing completion message and semantic_model_files
        """
        try:
            if not self.generation_evidence.validation_passed:
                return FuncToolResult(
                    success=0,
                    error=(
                        "validate_semantic must pass before publishing semantic models. "
                        "Call validate_semantic, fix any issues, and retry end_semantic_model_generation."
                    ),
                    result={"semantic_model_files": semantic_model_files},
                )

            logger.info(
                f"Semantic model generation completed for {len(semantic_model_files)} files: {semantic_model_files}"
            )
            self._semantic_object_exists_cache.clear()
            self._semantic_table_object_index = None

            return FuncToolResult(
                result={
                    "message": f"Semantic model generation completed for {len(semantic_model_files)} file(s)",
                    "semantic_model_files": semantic_model_files,
                }
            )

        except Exception as e:
            logger.error(f"Error completing semantic model generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

    def end_metric_generation(
        self,
        metric_file: str,
        semantic_model_files: Optional[List[str]] = None,
        metric_sqls_json: str = "",
    ) -> FuncToolResult:
        """
        Complete metric generation process and automatically sync to Knowledge Base.

        Call this tool when you have finished generating a metric YAML file.
        This tool automatically syncs the metric to the vector store (no user confirmation needed).

        Args:
            metric_file: Path to the generated metric YAML file (required).
                Relative paths (e.g. ``"metrics/orders_metrics.yml"``) are preferred
                and resolved against the sub-agent's semantic-model workspace using
                the live ``agent_config.current_datasource``. Absolute paths are only
                accepted when they resolve inside the Knowledge Base semantic-model
                sandbox.
            semantic_model_files: Paths to semantic model files that were newly
                created or updated and define the measure(s) used by this metric
                batch. Same relative/absolute rules as ``metric_file``.
            metric_sqls_json: JSON string mapping metric names to their generated SQL (from query_metrics dry_run).
                              Example: '{"revenue_total": "SELECT SUM(revenue) FROM orders GROUP BY date"}'

        Returns:
            dict: Result containing completion message, file paths, metric SQLs, and sync status
        """
        import json

        try:
            # Parse JSON string to dict
            metric_sqls: Dict[str, str] = {}
            if metric_sqls_json:
                try:
                    parsed = json.loads(metric_sqls_json)
                    if not isinstance(parsed, dict):
                        return FuncToolResult(success=0, error="metric_sqls_json must decode to a JSON object")
                    metric_sqls = parsed
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse metric_sqls_json: {e}")

            if not self.generation_evidence.validation_passed:
                return FuncToolResult(
                    success=0,
                    error=(
                        "validate_semantic must pass before publishing metrics. "
                        "Call validate_semantic, fix any issues, and retry end_metric_generation."
                    ),
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files or [],
                        "metric_sqls": metric_sqls,
                    },
                )

            if not self.generation_evidence.metric_dry_run_passed:
                return FuncToolResult(
                    success=0,
                    error=(
                        "query_metrics(dry_run=True) must pass before publishing metrics. "
                        "Run a dry-run query for the generated metric(s), fix any issues, and retry "
                        "end_metric_generation."
                    ),
                    result={
                        "metric_file": metric_file,
                        "semantic_model_files": semantic_model_files or [],
                        "metric_sqls": metric_sqls,
                    },
                )

            if self.generation_evidence.metric_sqls:
                metric_sqls = dict(self.generation_evidence.metric_sqls)

            logger.info(
                f"Metric generation completed: metric_file={metric_file}, "
                f"semantic_model_files={semantic_model_files or []}, "
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
            semantic_model_files = semantic_model_files or []
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
            for semantic_model_file in semantic_model_files:
                abs_semantic = _resolve(semantic_model_file, "semantic")
                if not abs_semantic:
                    return FuncToolResult(
                        success=0,
                        error=f"semantic_model_files contains path outside Knowledge Base sandbox: {semantic_model_file!r}",
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
            conflict_error = self._validate_metric_name_conflicts(metric_definitions)
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
            required_metric_names = self._metric_names_requiring_dry_run(metric_names, metric_definitions, metric_sqls)
            if required_metric_names and not self.generation_evidence.has_metric_dry_run(required_metric_names):
                return FuncToolResult(
                    success=0,
                    error=(
                        "query_metrics(dry_run=True) must pass for generated metric(s): "
                        f"{', '.join(required_metric_names)}. Run a dry-run query for these metric names, "
                        "fix any issues, and retry end_metric_generation."
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
            sync_result = self._sync_metric_to_db(abs_metric, abs_semantic_files, metric_sqls)

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

            self.generation_evidence.mark_kb_sync("metric")
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
                    "sync": sync_result,
                }
            )

        except Exception as e:
            logger.error(f"Error completing metric generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

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

        Batch generation often appends new metrics to an existing metrics YAML file.
        Requiring every historical metric in that file to be re-dry-run makes later
        batches slower and brittle. We still require all new metric names, and any
        metric name that the current run produced SQL evidence for.
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

    def _existing_metric_names(self) -> Optional[set[str]]:
        try:
            rows = self.metric_rag.search_all_metrics(select_fields=["name"])
        except Exception as exc:
            logger.warning("Failed to load existing metric names before publish dry-run gating: %s", exc)
            return None
        if not _is_supported_row_container(rows):
            return None
        normalized_rows = _rows_to_dicts(rows)
        names = set()
        for row in normalized_rows:
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
        normalized_rows = _rows_to_dicts(rows)

        for row in normalized_rows:
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

    def _sync_metric_to_db(
        self,
        metric_file: str,
        semantic_model_files: Optional[List[str]] = None,
        metric_sqls: Optional[Dict[str, str]] = None,
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

            result = GenerationHooks._sync_semantic_to_db(
                metric_file,
                self.agent_config,
                include_semantic_objects=False,
                include_metrics=True,
                metric_sqls=metric_sqls,
                original_yaml_path=metric_file,
            )
            if result.get("success"):
                result["semantic_synced"] = bool(synced_semantic_files)
                result["semantic_model_files_synced"] = synced_semantic_files

            if result.get("success"):
                logger.info(f"Successfully synced metric to KB: {result.get('message')}")
            else:
                logger.error(f"Failed to sync metric to KB: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Error syncing metric to KB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

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
