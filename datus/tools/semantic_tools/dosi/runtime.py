# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""The Agent-owned Dosi runtime: translates existing tool calls to the native engine.

All planning, SQL generation, and execution happen inside the Rust engine;
this class maps the Datus contract onto the engine API. Engine calls are
synchronous and GIL-releasing, so they run under ``asyncio.to_thread``.

Each native engine instance still serves one OSI model file. Directory-backed
runtimes maintain one engine handle per file and route globally unique metric
names to their owning model. Subject-tree ``path`` filters apply to discovery;
metric identity remains name-based.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from datus.tools.semantic_tools.authoring import MetricMutationResult, MetricSource
from datus.tools.semantic_tools.dosi.authoring import dosi_validation_text_payload
from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.dialects import resolve_engine_dialect
from datus.tools.semantic_tools.dosi.engine import (
    EngineHandle,
    EngineRegistry,
    datus_extension_version,
    load_binding,
)
from datus.tools.semantic_tools.dosi.errors import (
    SemanticValidationException,
    raise_mapped,
)
from datus.tools.semantic_tools.dosi.model import (
    iter_semantic_models,
    load_document,
    metric_payloads,
    queryable_grains,
)
from datus.tools.semantic_tools.exceptions import SemanticCoreException
from datus.tools.semantic_tools.models import (
    AttributionRequest,
    AttributionResult,
    DimensionInfo,
    MetricDefinition,
    QueryResult,
    SemanticModelInfo,
    SemanticValidationError,
    ValidationIssue,
    ValidationResult,
)


class DosiRuntime:
    """Agent's model catalog, query, validation, and editor boundary for Dosi."""

    def __init__(self, config: DosiConfig):
        self.service_type = "dosi"
        self.datasource = config.datasource
        self.config: DosiConfig = config
        self._registry = EngineRegistry(config)
        self._catalog_lock = threading.Lock()
        self._catalog_signature: tuple[tuple[str, int, int], ...] = ()
        self._catalog_handles: tuple[tuple[str, EngineHandle], ...] = ()
        self._metric_to_path: Dict[str, str] = {}
        self._model_to_path: Dict[str, str] = {}

    # ==================== Semantic Model Interface ====================

    def list_semantic_models(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> List[SemanticModelInfo]:
        handles, _, _ = self._catalog()
        return [self._model_info(row) for _, handle in handles for row in handle.get().datasets()]

    def get_semantic_model(
        self,
        table_name: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> Optional[SemanticModelInfo]:
        handles, _, _ = self._catalog()
        for _, handle in handles:
            for row in handle.get().datasets():
                source = str(row.get("source", ""))
                if table_name in (row.get("name"), source) or source.endswith(f".{table_name}"):
                    return self._model_info(row)
        return None

    # ==================== Metrics Interface ====================

    async def list_metrics(
        self,
        path: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MetricDefinition]:
        extension_version = await asyncio.to_thread(datus_extension_version)
        handles, _, _ = await asyncio.to_thread(self._catalog)
        binding = await asyncio.to_thread(load_binding)
        metrics = []
        for model_path, handle in handles:
            engine = await asyncio.to_thread(handle.get)
            rows = await asyncio.to_thread(engine.metrics)
            dataset_rows = await asyncio.to_thread(engine.datasets)
            dimension_rows = await asyncio.to_thread(engine.dimensions)
            payloads = await asyncio.to_thread(self._metric_payloads, model_path)
            connection = handle.profile_name
            dialect = self._dry_run_dialect(binding, connection)
            for row in rows:
                hints = payloads.get(str(row.get("name") or ""), {})
                metric_path = (
                    [str(item) for item in hints.get("subject_path")]
                    if isinstance(hints.get("subject_path"), list)
                    else None
                )
                if path and (metric_path is None or metric_path[: len(path)] != path):
                    continue
                window = hints.get("window")
                effective_time = self._effective_time_dimension(row, dataset_rows, dimension_rows)
                metadata: Dict[str, Any] = {
                    "datasets": list(row.get("datasets") or []),
                    "base_kind": row.get("kind"),
                    "datus_ext_version": extension_version,
                }
                for key in ("params", "param_schema"):
                    value = row.get(key)
                    if value:
                        metadata[key] = value
                # The D-DERIVE columns stop at the selection-sized scalars:
                # they let a caller tell a derived metric from its base, and
                # `derive_expr` — the authored formula — says how it is
                # assembled, which is what makes a composite decomposable.
                # `derive_members` cannot answer that: it folds coefficients
                # across inlined levels and keeps none for a non-linear
                # member, so a ratio reads as two bare metric names with
                # nothing saying they are divided. The structural columns
                # (derive_members, leaf_measures, conformed_dimensions,
                # attribution) stay off this listing; get_dimensions consumes
                # conformed_dimensions natively.
                for key in ("window_family", "window_function"):
                    if row.get(key):
                        metadata[key] = row[key]
                for key in (
                    "derive_family",
                    "derive_base",
                    "derive_expr",
                    "subset_of",
                ):
                    value = row.get(key)
                    if value is not None:
                        metadata[key] = value
                if effective_time:
                    metadata["time_dimension"] = effective_time
                if isinstance(window, dict):
                    requires_time_axis = await asyncio.to_thread(
                        self._probe_requires_time_axis,
                        engine,
                        binding,
                        str(row.get("name") or ""),
                        dialect=dialect,
                        connection=connection,
                    )
                    metadata.update(
                        {
                            "window": window,
                            "requires_time_axis": requires_time_axis,
                        }
                    )
                if "fill_nulls_with" in hints:
                    metadata["fill_nulls_with"] = hints["fill_nulls_with"]
                metrics.append(
                    MetricDefinition(
                        name=row["name"],
                        description=row.get("description") or None,
                        type=("window" if isinstance(window, dict) else row.get("kind")),
                        # Metric-level dimension discovery belongs to
                        # get_dimensions(), which delegates candidate selection and
                        # recommendation to Dosi's metric-scoped catalog.
                        dimensions=[],
                        measures=list(row.get("measures") or []),
                        unit=(str(hints.get("unit")) if hints.get("unit") is not None else None),
                        format=(str(hints.get("format")) if hints.get("format") is not None else None),
                        path=metric_path,
                        metadata=metadata,
                    )
                )
        return metrics[offset : offset + limit]

    async def get_dimensions(
        self,
        metric_name: str,
        path: Optional[List[str]] = None,
    ) -> List[DimensionInfo]:
        handle = await asyncio.to_thread(self._handle_for_metric, metric_name)
        engine = await asyncio.to_thread(handle.get)
        binding = await asyncio.to_thread(load_binding)
        # Dosi owns the metric-scoped candidate set, D-DIM recommendation,
        # conformance filtering, and ordering. Keep those semantics in one
        # place; this runtime only enriches native time rows with the grains that
        # compile for this metric and target.
        try:
            rows = await asyncio.to_thread(engine.dimensions, metric_name)
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors below
            raise_mapped(
                exc,
                binding,
                requested_metrics=[metric_name],
                requested_dimensions=[],
            )
            raise  # unreachable; raise_mapped always raises
        metric_rows = await asyncio.to_thread(engine.metrics)
        dataset_rows = await asyncio.to_thread(engine.datasets)
        metric_row = next(row for row in metric_rows if row.get("name") == metric_name)
        connection = handle.profile_name
        dialect = self._dry_run_dialect(binding, connection)

        requires_time_axis = await asyncio.to_thread(
            self._probe_requires_time_axis,
            engine,
            binding,
            metric_name,
            dialect=dialect,
            connection=connection,
        )
        physical_time_dimension = self._effective_time_dimension(
            metric_row,
            dataset_rows,
            [row for row in rows if row.get("name") != "metric_time"],
        )
        has_metric_time = any(row.get("name") == "metric_time" for row in rows)

        def _time_grains() -> Dict[str, List[str]]:
            supported: Dict[str, List[str]] = {}
            metric_time_error: Any = None
            for row in rows:
                name = str(row.get("name") or "")
                if not name or not row.get("is_time"):
                    continue
                grains: List[str] = []
                for grain in queryable_grains(row.get("time_granularity")):
                    group_by = [{"field": name, "grain": grain}]
                    if requires_time_axis and name not in {
                        "metric_time",
                        physical_time_dimension,
                    }:
                        group_by.append({"field": "metric_time", "grain": grain})
                    error = self._probe_compile(
                        engine,
                        binding,
                        metric_name,
                        group_by,
                        dialect=dialect,
                        connection=connection,
                    )
                    if error is None:
                        grains.append(grain)
                    elif name == "metric_time" and metric_time_error is None:
                        metric_time_error = error
                supported[name] = grains

            has_supported_axis = bool(
                supported.get("metric_time") or (not has_metric_time and supported.get(physical_time_dimension))
            )
            if requires_time_axis and not has_supported_axis:
                if metric_time_error is not None:
                    raise_mapped(
                        metric_time_error,
                        binding,
                        requested_metrics=[metric_name],
                        requested_dimensions=["metric_time"],
                    )
                raise SemanticValidationException(
                    SemanticValidationError(
                        code="no_primary_time_dimension",
                        metrics=[metric_name],
                        required_dimensions=["metric_time"],
                        message=(
                            f"metric {metric_name!r} requires a time axis, but "
                            "no supported metric_time grain was available"
                        ),
                    )
                )
            return supported

        grains_by_name = await asyncio.to_thread(_time_grains)
        return [
            DimensionInfo(
                name=str(row.get("name") or ""),
                description=row.get("description") or None,
                type="time" if row.get("is_time") else None,
                is_primary_time=bool(
                    row.get("name") == "metric_time" if has_metric_time else row.get("name") == physical_time_dimension
                ),
                time_granularities=(grains_by_name.get(str(row.get("name") or ""), []) if row.get("is_time") else []),
                recommended=row.get("is_dimension"),
                recommendation_source=(str(row.get("source")) if row.get("source") is not None else None),
            )
            for row in rows
            if row.get("name")
        ]

    async def lineage_graph(
        self,
        *,
        model: Optional[str] = None,
        redact_sql: bool = False,
    ) -> List[Dict[str, Any]]:
        """Metric lineage DAGs (dosi graph contract v2), one per model file.

        A pure projection of the compiled model — four lanes (physical table
        → dataset → atomic metric → derived metric), seven edge kinds; no SQL
        executes. ``model`` narrows the result to the file declaring that
        semantic model; ``redact_sql`` replaces every SQL text and opaque
        custom-extension ``data`` value in the payload with ``"<redacted>"``
        while keeping the graph shape intact.
        """
        handles, _, model_to_path = await asyncio.to_thread(self._catalog)
        if model is not None:
            path = model_to_path.get(model)
            if path is None:
                known = ", ".join(sorted(model_to_path)) or "<none>"
                raise SemanticCoreException(f"semantic model {model!r} not found (known models: {known})")
            handles = tuple((p, h) for p, h in handles if p == path)
        graphs: List[Dict[str, Any]] = []
        for _, handle in handles:
            engine = await asyncio.to_thread(handle.get)
            lineage = getattr(engine, "lineage", None)
            if lineage is None:
                raise SemanticCoreException(
                    "the installed dosi-engine binding predates the lineage API; upgrade to dosi-engine>=0.1.9"
                )
            graphs.append(await asyncio.to_thread(lineage, redact_sql=redact_sql))
        return graphs

    def artifact_metric_binding(self, metric_name: str) -> dict[str, str]:
        """Resolve an artifact's metric to its complete model and connection.

        Model names remain unique within a datasource. The full document hash
        pins datasets, relationships and derived members as well as the metric.
        """
        handles, metric_to_path, _ = self._catalog()
        handle = self._handle_for_metric(metric_name)
        path = metric_to_path[metric_name]
        return {
            "model_path": path,
            "model_revision": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "datasource": self.datasource or handle.profile_name or "default",
        }

    async def query_metrics(
        self,
        metrics: List[str],
        dimensions: Optional[List[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_granularity: Optional[str] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[List[str]] = None,
        dry_run: bool = False,
        *,
        params: Optional[Dict[str, Any]] = None,
        context_filter: Optional[str] = None,
    ) -> QueryResult:
        binding = await asyncio.to_thread(load_binding)
        handle = await asyncio.to_thread(self._handle_for_metrics, metrics)
        engine = await asyncio.to_thread(handle.get)
        dimensions = list(dimensions or [])
        # Fetched off the event loop; _build_query stays pure (no engine I/O).
        dimension_rows = await asyncio.to_thread(engine.dimensions)
        metric_rows = await asyncio.to_thread(engine.metrics)

        query = self._build_query(
            dimension_rows,
            metric_rows,
            metrics=metrics,
            dimensions=dimensions,
            time_start=time_start,
            time_end=time_end,
            time_granularity=time_granularity,
            where=where,
            limit=limit,
            order_by=order_by,
            params=params,
            context_filter=context_filter,
        )
        connection = handle.profile_name
        try:
            if dry_run:
                compiled = await asyncio.to_thread(
                    engine.compile,
                    query,
                    dialect=self._dry_run_dialect(binding, connection),
                    connection=connection,
                    # Dry-run SQL is read by humans and LLM callers, never fed
                    # to a driver, so hand back the formatted form.
                    pretty=True,
                )
                return QueryResult(
                    columns=["sql"],
                    data=[{"sql": compiled["sql"]}],
                    metadata={
                        "sql": compiled["sql"],
                        "dialect": compiled["dialect"],
                        "dry_run": True,
                        "explain": True,
                        **({"outputs": compiled["outputs"]} if compiled.get("outputs") is not None else {}),
                    },
                )
            result = await asyncio.to_thread(
                engine.execute,
                query,
                connection=connection,
                timeout_secs=float(self.config.timeout_seconds),
            )
            return QueryResult(
                columns=list(result["columns"]),
                data=list(result["rows"]),
                metadata={
                    "sql": result["sql"],
                    "dialect": result["dialect"],
                    "row_count": result["row_count"],
                    **({"outputs": result["outputs"]} if result.get("outputs") is not None else {}),
                },
            )
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors below
            # A SemanticValidationException (raised only by _build_query, before
            # this try) is not an engine error; raise_mapped re-raises it as-is.
            raise_mapped(
                exc,
                binding,
                requested_metrics=metrics,
                requested_dimensions=dimensions,
                requested_time_granularity=time_granularity,
            )
            raise  # unreachable; raise_mapped always raises

    async def attribute(self, request: AttributionRequest) -> AttributionResult:
        """Delegate attribution to the native engine that owns the metric."""
        dimensions = list(dict.fromkeys(dimension.strip() for dimension in request.dimensions if dimension.strip()))
        if not dimensions:
            raise SemanticValidationException(
                SemanticValidationError(
                    code="dimensions_required",
                    metrics=[request.metric],
                    message=(
                        "Attribution requires at least one candidate dimension; "
                        "call get_dimensions and retry with the selected dimensions."
                    ),
                )
            )
        binding = await asyncio.to_thread(load_binding)
        handle = await asyncio.to_thread(self._handle_for_metric, request.metric)
        engine = await asyncio.to_thread(handle.get)
        native_request = request.model_dump(mode="python", exclude_none=True)
        native_request["dimensions"] = dimensions
        try:
            result = await asyncio.to_thread(
                engine.attribute,
                native_request,
                connection=handle.profile_name,
                timeout_secs=float(self.config.timeout_seconds),
            )
            return AttributionResult.model_validate({**result, "implementation": "dosi"})
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors below
            raise_mapped(
                exc,
                binding,
                requested_metrics=[request.metric],
                requested_dimensions=dimensions,
            )
            raise  # unreachable; raise_mapped always raises

    async def validate_semantic(
        self,
        scope: str = "all",
        semantic_model_name: str = "",
        *,
        metric_names: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate one selected model or every model in the datasource."""
        if scope not in {"all", "semantic_model"}:
            return ValidationResult(
                valid=False,
                issues=[
                    ValidationIssue(
                        severity="error",
                        message="scope must be one of: all, semantic_model",
                    )
                ],
            )

        try:
            _, handles = self._registry.snapshot()
        except SemanticCoreException as exc:
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(severity="error", message=str(exc))],
            )

        all_model_paths = [path for path, _ in handles]
        model_paths = list(all_model_paths)
        if semantic_model_name:
            selected, available = self._find_model_paths(model_paths, semantic_model_name)
            if len(model_paths) == 1:
                selected = model_paths
            elif not selected:
                return ValidationResult(
                    valid=False,
                    issues=[
                        ValidationIssue(
                            severity="error",
                            message=(
                                "semantic_model_not_found: semantic model "
                                f"{semantic_model_name!r} was not found; "
                                f"available: {', '.join(available) or '(none)'}"
                            ),
                            location="semantic_model",
                        )
                    ],
                )
            elif len(selected) > 1:
                return ValidationResult(
                    valid=False,
                    issues=[
                        ValidationIssue(
                            severity="error",
                            message=(
                                f"semantic model {semantic_model_name!r} is declared "
                                f"in multiple files: {', '.join(selected)}"
                            ),
                            location="semantic_model",
                        )
                    ],
                )
            model_paths = selected
        elif scope == "semantic_model" and len(model_paths) > 1:
            _, available = self._find_model_paths(model_paths, "")
            return ValidationResult(
                valid=False,
                issues=[
                    ValidationIssue(
                        severity="error",
                        message=(
                            "semantic_model_name is required for targeted validation "
                            "when multiple models exist; available: "
                            f"{', '.join(available) or '(none)'}"
                        ),
                        location="semantic_model",
                    )
                ],
            )

        def _validate(model_path: str) -> tuple[dict[str, Any], str]:
            with open(model_path, encoding="utf-8") as fh:
                model_text = fh.read()
            payload = dosi_validation_text_payload(
                model_text,
                metric_names=metric_names,
            )
            if payload.get("valid") and semantic_model_name:
                document = yaml.safe_load(model_text) or {}
                names = (
                    [
                        str(model.get("name") or "")
                        for model in document.get("semantic_model", [])
                        if isinstance(model, dict)
                    ]
                    if isinstance(document, dict)
                    else []
                )
                if semantic_model_name not in names:
                    payload["valid"] = False
                    payload.setdefault("issues", []).append(
                        {
                            "severity": "error",
                            "code": "semantic_model_not_found",
                            "location": "semantic_model",
                            "message": (
                                f"semantic model {semantic_model_name!r} "
                                "was not found; "
                                f"available: {', '.join(names) or '(none)'}"
                            ),
                        }
                    )
            artifact_digest = "sha256:" + hashlib.sha256(model_text.encode("utf-8")).hexdigest()
            return payload, artifact_digest

        issues: List[ValidationIssue] = []
        valid = True
        qualify = len(model_paths) > 1
        contract_digests: set[str] = set()
        compiled_metrics: List[Dict[str, Any]] = []
        compiled_metric_digests: Dict[str, str] = {}
        artifact_digests: Dict[str, str] = {}
        for model_path in model_paths:
            try:
                payload, artifact_digest = await asyncio.to_thread(_validate, model_path)
            except OSError as exc:
                valid = False
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"cannot read semantic model {model_path!r}: {exc}",
                    )
                )
                continue

            artifact_digests[str(Path(model_path).resolve())] = artifact_digest
            valid = valid and bool(payload.get("valid"))
            contract_digest = str(payload.get("contract_digest") or "").strip()
            if contract_digest:
                contract_digests.add(contract_digest)
            if payload.get("valid"):
                compiled_metrics.extend(payload.get("compiled_metrics") or [])
                compiled_metric_digests.update(payload.get("compiled_metric_digests") or {})
            prefix = f"[semantic_model_file={Path(model_path).name}] " if qualify else ""
            for issue in payload.get("issues", []):
                issues.append(
                    ValidationIssue(
                        severity=issue.get("severity") or "error",
                        message=(f"{prefix}{issue.get('code', 'issue')}: {issue.get('message', '')}"),
                        location=issue.get("location") or None,
                    )
                )
            for err in payload.get("compile_errors", []):
                message = f"{prefix}{err.get('code', 'compile_error')}: {err.get('message', '')}"
                if err.get("hint"):
                    message = f"{message} | {err['hint']}"
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=message,
                        location=err.get("location") or None,
                    )
                )

        relevant_identity_paths = set(model_paths) if semantic_model_name or scope == "semantic_model" else None
        identity_issues = await asyncio.to_thread(
            self._cross_file_identity_issues,
            all_model_paths,
            relevant_paths=relevant_identity_paths,
        )
        issues.extend(identity_issues)
        valid = valid and not identity_issues
        if len(contract_digests) > 1:
            valid = False
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=("engine_contract_mismatch: validation returned multiple authoring contract digests"),
                )
            )

        requested_metrics = list(dict.fromkeys(metric_names or []))
        compiled_names = {str(metric.get("name") or "") for metric in compiled_metrics if isinstance(metric, dict)}
        missing_metrics = [name for name in requested_metrics if name not in compiled_names]
        if missing_metrics:
            valid = False
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        "compiled_metric_not_found: validation did not compile "
                        f"requested metrics: {', '.join(missing_metrics)}"
                    ),
                    location="metrics",
                )
            )

        metadata: Dict[str, Any] = {}
        if valid:
            metadata = {
                "contract_digest": next(iter(contract_digests), ""),
                "compiled_metrics": compiled_metrics,
                "compiled_metric_digests": compiled_metric_digests,
                "artifact_sha256": artifact_digests,
                "metric_names": requested_metrics,
            }
        return ValidationResult(valid=valid, issues=issues, metadata=metadata)

    # ==================== Authoring Interface ====================
    # Explorer/editor surface; not an agent/LLM tool. The source of truth is
    # the OSI YAML file, and the native engine validates each mutation.

    def _authoring_root(self) -> str:
        """The OSI model path authoring operates on (mirrors resolve_model_file).

        An explicit ``semantic_model_path`` pins authoring to exactly that file
        so sibling models in the same directory are never touched; otherwise the
        configured ``semantic_models_path`` directory is scanned.
        """
        if self.config.semantic_model_path:
            return self.config.semantic_model_path
        if self.config.semantic_models_path:
            return self.config.semantic_models_path
        raise SemanticCoreException("dosi authoring requires semantic_model_path or semantic_models_path")

    def _author(self) -> Any:
        from datus.tools.semantic_tools.dosi.authoring import DosiMetricAuthor

        return DosiMetricAuthor(self._authoring_root())

    def read_metric_source(
        self,
        metric_name: str,
        *,
        subject_path: Optional[List[str]] = None,
    ) -> MetricSource:
        return self._author().read(metric_name)

    def write_metric_source(
        self,
        metric_name: str,
        source: str,
        *,
        subject_path: Optional[List[str]] = None,
        create: bool = False,
    ) -> MetricMutationResult:
        return self._author().write(metric_name, source, subject_path=subject_path, create=create)

    def delete_metric_source(
        self,
        metric_name: str,
        *,
        subject_path: Optional[List[str]] = None,
    ) -> MetricMutationResult:
        return self._author().delete(metric_name)

    def restore_metric_source(self, source: MetricSource) -> MetricMutationResult:
        """Restore a deleted metric to its original model file."""
        from datus.tools.semantic_tools.dosi.authoring import DosiMetricAuthor

        return DosiMetricAuthor(source.file_path).write(source.name, source.text, create=True)

    # ==================== Internals ====================

    @staticmethod
    def _document_model_names(model_path: str) -> List[str]:
        return [
            str(model.get("name") or "").strip()
            for model in iter_semantic_models(load_document(model_path))
            if str(model.get("name") or "").strip()
        ]

    @classmethod
    def _find_model_paths(cls, model_paths: List[str], semantic_model_name: str) -> tuple[List[str], List[str]]:
        matches: List[str] = []
        available: List[str] = []
        for model_path in model_paths:
            try:
                names = cls._document_model_names(model_path)
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
            available.extend(names)
            if semantic_model_name and semantic_model_name in names:
                matches.append(model_path)
        return matches, sorted(set(available))

    @classmethod
    def _cross_file_identity_issues(
        cls,
        model_paths: List[str],
        *,
        relevant_paths: Optional[set[str]] = None,
    ) -> List[ValidationIssue]:
        """Return global identity conflicts relevant to the validation scope.

        Full validation reports every cross-file conflict. Targeted validation
        passes the selected file paths so it rejects conflicts involving the
        target without surfacing unrelated conflicts between sibling models.
        """
        model_owners: Dict[str, str] = {}
        metric_owners: Dict[str, str] = {}
        issues: List[ValidationIssue] = []
        for model_path in model_paths:
            try:
                document = load_document(model_path)
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
            for model in iter_semantic_models(document):
                model_name = str(model.get("name") or "").strip()
                if model_name:
                    owner = model_owners.get(model_name)
                    if owner and owner != model_path:
                        if relevant_paths is None or relevant_paths.intersection({owner, model_path}):
                            issues.append(
                                ValidationIssue(
                                    severity="error",
                                    message=(
                                        "duplicate_semantic_model: "
                                        f"{model_name!r} is declared in {owner!r} "
                                        f"and {model_path!r}"
                                    ),
                                    location="semantic_model",
                                )
                            )
                    else:
                        model_owners[model_name] = model_path
                for metric in model.get("metrics") or []:
                    if not isinstance(metric, dict):
                        continue
                    metric_name = str(metric.get("name") or "").strip()
                    if not metric_name:
                        continue
                    owner = metric_owners.get(metric_name)
                    if owner and owner != model_path:
                        if relevant_paths is None or relevant_paths.intersection({owner, model_path}):
                            issues.append(
                                ValidationIssue(
                                    severity="error",
                                    message=(
                                        f"duplicate_metric: {metric_name!r} is "
                                        f"declared in {owner!r} and {model_path!r}; "
                                        "metric names must be unique within a "
                                        "datasource"
                                    ),
                                    location="metrics",
                                )
                            )
                    else:
                        metric_owners[metric_name] = model_path
        return issues

    def _catalog(
        self,
    ) -> tuple[tuple[tuple[str, EngineHandle], ...], Dict[str, str], Dict[str, str]]:
        with self._catalog_lock:
            signature, handles = self._registry.snapshot()
            if signature == self._catalog_signature:
                return (
                    self._catalog_handles,
                    dict(self._metric_to_path),
                    dict(self._model_to_path),
                )

            metric_to_path: Dict[str, str] = {}
            model_to_path: Dict[str, str] = {}
            for model_path, handle in handles:
                try:
                    model_names = self._document_model_names(model_path)
                except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
                    raise SemanticCoreException(f"cannot read semantic model {model_path!r}: {exc}") from exc
                for model_name in model_names:
                    owner = model_to_path.get(model_name)
                    if owner and owner != model_path:
                        raise SemanticCoreException(
                            f"semantic model {model_name!r} is declared in {owner!r} and {model_path!r}"
                        )
                    model_to_path[model_name] = model_path
                for row in handle.get().metrics():
                    metric_name = str(row.get("name") or "").strip()
                    if not metric_name:
                        continue
                    owner = metric_to_path.get(metric_name)
                    if owner and owner != model_path:
                        raise SemanticCoreException(
                            f"metric {metric_name!r} is declared in {owner!r} "
                            f"and {model_path!r}; metric names must be unique "
                            "within a datasource"
                        )
                    metric_to_path[metric_name] = model_path

            self._catalog_signature = signature
            self._catalog_handles = handles
            self._metric_to_path = metric_to_path
            self._model_to_path = model_to_path
            return handles, dict(metric_to_path), dict(model_to_path)

    @staticmethod
    def _model_label(model_path: str, model_to_path: Dict[str, str]) -> str:
        names = sorted(name for name, owner in model_to_path.items() if owner == model_path)
        return names[0] if names else Path(model_path).name

    def _handle_for_metric(self, metric_name: str) -> EngineHandle:
        handles, metric_to_path, _ = self._catalog()
        model_path = metric_to_path.get(metric_name)
        if model_path is None:
            candidates = ", ".join(sorted(metric_to_path))
            raise SemanticValidationException(
                SemanticValidationError(
                    code="unknown_metric",
                    metrics=[metric_name],
                    message=(f"unknown metric {metric_name!r} | candidates: {candidates or '(none)'}"),
                )
            )
        return dict(handles)[model_path]

    def _handle_for_metrics(self, metrics: List[str]) -> EngineHandle:
        handles, metric_to_path, model_to_path = self._catalog()
        unknown = [name for name in dict.fromkeys(metrics) if name not in metric_to_path]
        if unknown:
            candidates = ", ".join(sorted(metric_to_path))
            raise SemanticValidationException(
                SemanticValidationError(
                    code="unknown_metric",
                    metrics=unknown,
                    message=(f"unknown metric(s): {', '.join(unknown)} | candidates: {candidates or '(none)'}"),
                )
            )
        if not metrics:
            raise SemanticValidationException(
                SemanticValidationError(
                    code="empty_query",
                    metrics=[],
                    message="at least one metric is required",
                )
            )
        owners = {metric_to_path[name] for name in metrics}
        if len(owners) > 1:
            details = ", ".join(
                f"{name} -> {self._model_label(metric_to_path[name], model_to_path)}" for name in dict.fromkeys(metrics)
            )
            raise SemanticValidationException(
                SemanticValidationError(
                    code="cross_semantic_model_query_unsupported",
                    metrics=list(dict.fromkeys(metrics)),
                    message=(
                        f"metrics belong to multiple semantic models: {details}; "
                        "cross-semantic-model queries are not supported"
                    ),
                )
            )
        return dict(handles)[next(iter(owners))]

    def _model_info(self, row: Dict[str, Any]) -> SemanticModelInfo:
        return SemanticModelInfo(
            name=str(row.get("name", "")),
            table_name=str(row.get("source", "")),
            platform_type="dosi",
            extra={k: v for k, v in row.items() if k not in ("name", "source")},
        )

    @staticmethod
    def _metric_payloads(model_path: str) -> Dict[str, Dict[str, Any]]:
        """Presentation metadata from the raw model; Dosi still validates it."""

        return metric_payloads(load_document(model_path))

    @staticmethod
    def _probe_compile(
        engine: Any,
        binding: Any,
        metric_name: str,
        group_by: List[Dict[str, Any]],
        *,
        dialect: Optional[str],
        connection: Optional[str],
    ) -> Any:
        """Compile one discovery query and return only planner rejections."""

        try:
            engine.compile(
                {"metrics": [metric_name], "group_by": group_by},
                dialect=dialect,
                connection=connection,
            )
        except binding.QueryError as exc:
            return exc
        except Exception as exc:  # noqa: BLE001 - re-raised by raise_mapped
            raise_mapped(
                exc,
                binding,
                requested_metrics=[metric_name],
                requested_dimensions=[str(item.get("field") or "") for item in group_by],
            )
        return None

    @classmethod
    def _probe_requires_time_axis(
        cls,
        engine: Any,
        binding: Any,
        metric_name: str,
        *,
        dialect: Optional[str],
        connection: Optional[str],
    ) -> bool:
        """Ask the native planner whether a metric needs a grouped time axis."""

        without_axis = cls._probe_compile(
            engine,
            binding,
            metric_name,
            [],
            dialect=dialect,
            connection=connection,
        )
        if without_axis is None:
            return False
        for grain in queryable_grains(None):
            if (
                cls._probe_compile(
                    engine,
                    binding,
                    metric_name,
                    [{"field": "metric_time", "grain": grain}],
                    dialect=dialect,
                    connection=connection,
                )
                is None
            ):
                return True
        return str(getattr(without_axis, "code", "")) in {
            "no_primary_time_dimension",
            "unknown_dimension",
        }

    @staticmethod
    def _effective_time_dimension(
        metric_row: Dict[str, Any],
        dataset_rows: List[Dict[str, Any]],
        dimension_rows: List[Dict[str, Any]],
    ) -> str:
        """Discover time dimensions from the engine's compiled catalog rows."""

        dimension_names = {str(row.get("name") or "") for row in dimension_rows if row.get("name")}
        explicit = str(metric_row.get("time_dimension") or "").strip()
        if explicit:
            if explicit in dimension_names:
                return explicit
            metric_datasets = [str(name) for name in metric_row.get("datasets") or [] if name]
            qualified = [
                f"{dataset}.{explicit}" for dataset in metric_datasets if f"{dataset}.{explicit}" in dimension_names
            ]
            if len(qualified) == 1:
                return qualified[0]

        metric_datasets = {str(name) for name in metric_row.get("datasets") or [] if name}
        candidates: set[str] = set()
        for dataset in dataset_rows:
            dataset_name = str(dataset.get("name") or "")
            if dataset_name not in metric_datasets:
                continue
            primary = str(dataset.get("primary_time_dimension") or "").strip()
            if not primary:
                time_dimensions = [str(name) for name in dataset.get("time_dimensions") or [] if name]
                if len(time_dimensions) == 1:
                    primary = time_dimensions[0]
            if primary:
                candidate = primary if "." in primary else f"{dataset_name}.{primary}"
                if candidate in dimension_names:
                    candidates.add(candidate)
        return next(iter(candidates)) if len(candidates) == 1 else ""

    def _dry_run_dialect(self, binding: Any, connection: Optional[str]) -> Optional[str]:
        """Dialect for compile-only calls: explicit config, else db_config type.

        With a connection profile the engine already knows the dialect; an
        agreeing explicit dialect is harmless, a conflicting one is a config
        error the engine reports. An explicitly configured dialect that the
        engine doesn't know is a config error (raise) — not silently dropped,
        which would emit DuckDB SQL for the wrong dialect. An unknown
        db_config *type* stays lenient (returns None → engine decides).
        """
        if self.config.dialect:
            resolved = resolve_engine_dialect(self.config.dialect, binding.DIALECTS)
            if resolved is None:
                raise SemanticCoreException(
                    f"unknown dialect {self.config.dialect!r}; supported: {', '.join(binding.DIALECTS)}"
                )
            return resolved
        db_config = self.config.db_config or {}
        return resolve_engine_dialect(db_config.get("type"), binding.DIALECTS)

    def _build_query(
        self,
        dimension_rows: List[Dict[str, Any]],
        metric_rows: List[Dict[str, Any]],
        *,
        metrics: List[str],
        dimensions: List[str],
        time_start: Optional[str],
        time_end: Optional[str],
        time_granularity: Optional[str],
        where: Optional[str],
        limit: Optional[int],
        order_by: Optional[List[str]],
        params: Optional[Dict[str, Any]] = None,
        context_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble the engine's MetricQuery dict. Pure: no engine I/O."""
        del metric_rows  # Kept for backward-compatible direct-call tests/callers.
        time_dimension_names = {row["name"] for row in dimension_rows if row.get("is_time")}

        def is_time_dimension(name: str) -> bool:
            return (
                name == "metric_time"
                or name in time_dimension_names
                or any(full.endswith(f".{name}") for full in time_dimension_names)
            )

        group_by: List[Dict[str, Any]] = []
        grain_attached = False
        requested_grain = str(time_granularity or "").strip().lower() or None
        for dimension in dimensions:
            item: Dict[str, Any] = {"field": dimension}
            if requested_grain and is_time_dimension(dimension):
                item["grain"] = requested_grain
                grain_attached = True
            group_by.append(item)

        if time_granularity and not grain_attached:
            model_time_dims = ["metric_time", *sorted(time_dimension_names)]
            raise SemanticValidationException(
                SemanticValidationError(
                    code="time_grain_required",
                    metrics=list(metrics),
                    required_dimensions=model_time_dims,
                    required_time_granularity=time_granularity,
                    suggested_retry=(
                        {
                            "metrics": list(metrics),
                            "dimensions": dimensions + model_time_dims[:1],
                            "time_granularity": time_granularity,
                        }
                        if model_time_dims
                        else None
                    ),
                    message=(
                        "time_granularity was given but no requested dimension is a "
                        "time dimension | time dimensions: "
                        f"{', '.join(model_time_dims)}"
                    ),
                )
            )

        query: Dict[str, Any] = {"metrics": list(metrics), "group_by": group_by}
        if params:
            query["params"] = dict(params)
        if where:
            query["where_sql"] = where
        if context_filter:
            query["context_filter"] = context_filter
        if time_start or time_end:
            time_range: Dict[str, Any] = {"start": time_start, "end": time_end}
            if not any(is_time_dimension(item["field"]) for item in group_by):
                # The reserved time name lets the engine resolve each metric's
                # effective axis and report no_primary_time_dimension or
                # metric_time_conflict itself. The Agent must not guess from
                # raw dataset membership.
                time_range["dimension"] = "metric_time"
            query["time_range"] = time_range
        if order_by:
            query["order_by"] = [
                {"key": key[1:], "desc": True} if key.startswith("-") else {"key": key, "desc": False}
                for key in order_by
            ]
        if limit is not None:
            query["limit"] = int(limit)
        return query
