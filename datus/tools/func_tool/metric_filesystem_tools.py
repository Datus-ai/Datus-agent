# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import copy
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from datus.storage.metric.store import normalize_metric_name
from datus.storage.semantic_model.artifact_file import atomic_write_text, semantic_artifact_lock
from datus.tools.func_tool.base import FuncToolResult, tool_schema
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.fs_path_policy import PathZone, ResolvedPath
from datus.tools.semantic_tools.metric_identity import (
    ensure_metric_uid,
    metric_definitions,
    metric_uid,
    prepare_metric_uids,
    validate_metric_uids,
)
from datus.utils.memory_loader import apply_single_replacement

if TYPE_CHECKING:
    from datus.tools.func_tool.generation_evidence import GenerationEvidence
    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState


class MetricFilesystemFuncTool(FilesystemFuncTool):
    """Filesystem tool variant for OSI semantic-model and metric authoring.

    Exposes narrow upsert/delete tools for metrics and datasets so authoring
    can update definitions without rewriting relationships or model metadata.
    """

    def __init__(
        self,
        *args,
        osi_target_state: Optional["OsiSemanticModelTargetState"] = None,
        generation_evidence: Optional["GenerationEvidence"] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.osi_target_state = osi_target_state
        self.generation_evidence = generation_evidence

    def _require_metric_revision(self, path: str | Path) -> None:
        self.osi_target_state.require_current_revision(path)

    def available_tools(self):
        """Expose the narrow OSI metric and dataset mutation tools."""
        from datus.tools.func_tool import trans_to_function_tool

        return [
            trans_to_function_tool(self.read_file),
            trans_to_function_tool(self.read_image),
            trans_to_function_tool(self.upsert_osi_metrics),
            trans_to_function_tool(self.delete_osi_metrics),
            trans_to_function_tool(self.upsert_osi_datasets),
            trans_to_function_tool(self.delete_osi_datasets),
            trans_to_function_tool(self.glob),
            trans_to_function_tool(self.grep),
        ]

    @staticmethod
    def all_tools_name() -> List[str]:
        """Return the complete conditional tool surface for permission routing."""
        names = FilesystemFuncTool.all_tools_name()
        for name in (
            "upsert_osi_metrics",
            "delete_osi_metrics",
            "upsert_osi_datasets",
            "delete_osi_datasets",
        ):
            if name not in names:
                names.append(name)
        return names

    @tool_schema(strict_mode=False)
    def upsert_osi_datasets(self, path: str, datasets_json: List[Dict[str, Any]] | str) -> FuncToolResult:
        """Create or update datasets in a planned OSI semantic-model file.

        Pass datasets_json as a non-empty array of complete OSI dataset objects,
        not an encoded string. Legacy JSON-encoded array strings remain accepted.
        By default,
        ``source`` is a qualified physical table or view; this does not require
        a primary or unique key. Follow the authoring skill's source-selection
        rules for a specific native capability gap or an explicit user request
        for a query-backed source. Such datasets provide a complete SELECT in
        ``source``; the DATUS query-source extension is added automatically.
        A missing planned file is created with the first non-empty dataset batch.
        Existing datasets are replaced by ``name`` and new datasets are appended.
        Relationships, metrics, and model metadata are preserved.

        Args:
            path: Existing, planned OSI semantic-model YAML file.
            datasets_json: Non-empty array of OSI dataset objects. Prefer a structured
                array; a JSON-encoded array string is accepted for legacy callers.
        """
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error
        target_path = resolved.resolved

        try:
            incoming_datasets = (
                json.loads(datasets_json) if isinstance(datasets_json, str) else copy.deepcopy(datasets_json)
            )
        except (TypeError, json.JSONDecodeError) as exc:
            return FuncToolResult(success=0, error=f"datasets_json is not a valid JSON-encoded array string: {exc}")
        if not isinstance(incoming_datasets, list) or not incoming_datasets:
            return FuncToolResult(success=0, error="datasets_json must be a non-empty JSON array")

        incoming_by_name: Dict[str, Dict[str, Any]] = {}
        for index, dataset in enumerate(incoming_datasets):
            if not isinstance(dataset, dict):
                return FuncToolResult(success=0, error=f"datasets_json[{index}] must be a JSON object")
            dataset = dict(dataset)
            source = str(dataset.get("source") or "").lstrip().lower()
            if re.match(r"^(?:select|with)\s", source):
                dataset["custom_extensions"] = self._query_source_extensions(dataset.get("custom_extensions"))
            name = str(dataset.get("name") or "").strip()
            if not name:
                return FuncToolResult(success=0, error=f"datasets_json[{index}].name is required")
            if name in incoming_by_name:
                return FuncToolResult(success=0, error=f"datasets_json contains duplicate dataset name: {name}")
            incoming_by_name[name] = dataset

        with semantic_artifact_lock(target_path):
            guard_error = self._mutation_guard_error(target_path)
            if guard_error is not None:
                return guard_error
            creating = not target_path.exists()
            original_content = b""
            if creating:
                planned = self.osi_target_state.planned if self.osi_target_state is not None else None
                planned_name = str((planned or {}).get("semantic_model_name") or "").strip()
                if not planned_name:
                    return FuncToolResult(
                        success=0,
                        error="Plan the OSI semantic-model target before creating its first dataset.",
                        result={"code": "semantic_model_target_required"},
                    )
                document = {
                    "version": "0.2.0.dev0",
                    "semantic_model": [
                        {
                            "name": planned_name,
                            "datasets": [],
                            "relationships": [],
                            "metrics": [],
                        }
                    ],
                }
            else:
                if not target_path.is_file():
                    return FuncToolResult(
                        success=0,
                        error=f"OSI semantic model path is not a file: {resolved.display}",
                    )
                try:
                    original_content = target_path.read_bytes()
                    document = yaml.safe_load(original_content.decode("utf-8"))
                except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                    return FuncToolResult(success=0, error=f"Cannot load OSI semantic model {resolved.display}: {exc}")

            if not isinstance(document, dict):
                return FuncToolResult(success=0, error="OSI semantic model root must be a YAML object")
            models = document.get("semantic_model")
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
                return FuncToolResult(success=0, error="OSI document must contain exactly one semantic_model object")

            model = models[0]
            existing_datasets = model["datasets"] if "datasets" in model else []
            if not isinstance(existing_datasets, list) or any(
                not isinstance(dataset, dict) for dataset in existing_datasets
            ):
                return FuncToolResult(success=0, error="semantic_model[0].datasets must be a list of dataset objects")

            dataset_indexes = {
                str(dataset.get("name") or "").strip(): index
                for index, dataset in enumerate(existing_datasets)
                if str(dataset.get("name") or "").strip()
            }
            created: List[str] = []
            updated: List[str] = []
            unchanged: List[str] = []
            for name, dataset in incoming_by_name.items():
                if name in dataset_indexes:
                    index = dataset_indexes[name]
                    if existing_datasets[index] == dataset:
                        unchanged.append(name)
                    else:
                        existing_datasets[index] = dataset
                        updated.append(name)
                else:
                    dataset_indexes[name] = len(existing_datasets)
                    existing_datasets.append(dataset)
                    created.append(name)

            if created or updated:
                model["datasets"] = existing_datasets
                validation_error = self._validate_osi_document(document)
                if validation_error:
                    return FuncToolResult(
                        success=0,
                        error=f"Invalid OSI dataset update: {validation_error}",
                        result={"written": False, "semantic_model_file": resolved.display},
                    )
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    if self.osi_target_state is not None:
                        self.osi_target_state.record_artifact_snapshot(
                            target_path,
                            original_content,
                            existed=not creating,
                        )
                    atomic_write_text(target_path, serialized)
                except OSError as exc:
                    return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
                self._notify_mutation(target_path)
                serialized_content = serialized.encode("utf-8")
            else:
                serialized_content = original_content
            if self.osi_target_state is not None and self.osi_target_state.bound is not None:
                self.osi_target_state.record_dataset_touch(
                    target_path,
                    serialized_content,
                    list(incoming_by_name),
                )
            elif self.osi_target_state is not None and self.osi_target_state.planned is not None:
                self.osi_target_state.record_planned_dataset_touch([*created, *updated])

        return FuncToolResult(
            result={
                "message": f"Upserted {len(incoming_by_name)} OSI dataset(s)",
                "semantic_model_file": resolved.display,
                "created": created,
                "updated": updated,
                "unchanged": unchanged,
            }
        )

    def delete_osi_datasets(self, path: str, dataset_names: List[str]) -> FuncToolResult:
        """Delete explicitly named datasets from a planned OSI semantic model.

        Only ``semantic_model[0].datasets`` is changed. Relationships, metrics,
        and model metadata are preserved. Resolve live references first: the
        complete candidate must validate before deletion is written. Missing
        names are successful no-ops, allowing retries to republish the final
        YAML and clean stale Knowledge Base rows.

        Args:
            path: Existing, planned OSI semantic-model YAML file.
            dataset_names: Dataset business names to remove.
        """
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error
        target_path = resolved.resolved

        if not isinstance(dataset_names, list):
            return FuncToolResult(success=0, error="dataset_names must be a JSON array of dataset names")
        requested: List[str] = []
        requested_keys: set[str] = set()
        for value in dataset_names:
            name = str(value or "").strip()
            key = name.casefold()
            if name and key not in requested_keys:
                requested.append(name)
                requested_keys.add(key)
        if not requested:
            return FuncToolResult(success=0, error="dataset_names must contain at least one non-empty name")

        with semantic_artifact_lock(target_path):
            guard_error = self._mutation_guard_error(target_path)
            if guard_error is not None:
                return guard_error
            if not target_path.exists() or not target_path.is_file():
                return FuncToolResult(
                    success=0,
                    error=f"OSI semantic model file not found: {resolved.display}",
                    result={"code": "semantic_model_required", "semantic_model_file": resolved.display},
                )
            try:
                original_content = target_path.read_bytes()
                document = yaml.safe_load(original_content.decode("utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                return FuncToolResult(success=0, error=f"Cannot load OSI semantic model {resolved.display}: {exc}")

            if not isinstance(document, dict):
                return FuncToolResult(success=0, error="OSI semantic model root must be a YAML object")
            models = document.get("semantic_model")
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
                return FuncToolResult(success=0, error="OSI document must contain exactly one semantic_model object")

            model = models[0]
            existing_datasets = model["datasets"] if "datasets" in model else []
            if not isinstance(existing_datasets, list) or any(
                not isinstance(dataset, dict) for dataset in existing_datasets
            ):
                return FuncToolResult(success=0, error="semantic_model[0].datasets must be a list of dataset objects")

            deleted = [
                str(dataset.get("name") or "").strip()
                for dataset in existing_datasets
                if str(dataset.get("name") or "").strip().casefold() in requested_keys
            ]
            deleted_keys = {name.casefold() for name in deleted}
            remaining_datasets = [
                dataset
                for dataset in existing_datasets
                if str(dataset.get("name") or "").strip().casefold() not in requested_keys
            ]
            already_absent = [name for name in requested if name.casefold() not in deleted_keys]

            if deleted:
                model["datasets"] = remaining_datasets
                validation_error = self._validate_osi_document(document)
                if validation_error:
                    return FuncToolResult(
                        success=0,
                        error=f"Invalid OSI dataset deletion: {validation_error}",
                        result={"written": False, "semantic_model_file": resolved.display},
                    )
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    if self.osi_target_state is not None:
                        self.osi_target_state.record_artifact_snapshot(target_path, original_content)
                    atomic_write_text(target_path, serialized)
                except OSError as exc:
                    return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
                serialized_content = serialized.encode("utf-8")
            else:
                serialized_content = original_content

            # A byte-preserving retry must still invalidate publication
            # evidence so stale semantic rows can be reconciled from the YAML.
            self._notify_mutation(target_path)
            if self.osi_target_state is not None and self.osi_target_state.bound is not None:
                self.osi_target_state.record_dataset_touch(target_path, serialized_content, requested)

        return FuncToolResult(
            result={
                "message": f"Processed deletion of {len(requested)} OSI dataset(s)",
                "semantic_model_file": resolved.display,
                "requested": requested,
                "deleted": deleted,
                "already_absent": already_absent,
                "remaining": [
                    str(dataset.get("name") or "").strip()
                    for dataset in remaining_datasets
                    if str(dataset.get("name") or "").strip()
                ],
            }
        )

    @staticmethod
    def _query_source_extensions(value: Any) -> List[Dict[str, Any]]:
        extensions = list(value) if isinstance(value, list) else ([value] if isinstance(value, dict) else [])
        for extension in extensions:
            if not isinstance(extension, dict) or str(extension.get("vendor_name") or "").upper() != "DATUS":
                continue
            data = extension.get("data")
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    parsed = {}
                parsed = parsed if isinstance(parsed, dict) else {}
                parsed["source_type"] = "query"
                extension["data"] = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            else:
                parsed = dict(data) if isinstance(data, dict) else {}
                parsed["source_type"] = "query"
                extension["data"] = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            return extensions
        extensions.append(
            {
                "vendor_name": "DATUS",
                "data": json.dumps({"source_type": "query"}, sort_keys=True),
            }
        )
        return extensions

    @tool_schema(strict_mode=False)
    def upsert_osi_metrics(self, path: str, metrics_json: List[Dict[str, Any]] | str) -> FuncToolResult:
        """Create or update metrics in an existing OSI semantic-model file.

        Pass metrics_json as a non-empty array of complete OSI metric objects,
        not an encoded string. Legacy JSON-encoded array strings remain accepted.
        Existing metrics are
        replaced by ``name`` and new metrics are appended. Identical metrics
        leave the file bytes unchanged but still enter this request's exact
        publish scope. The tool only owns the ``metrics`` collection, so
        datasets, fields, relationships, and model metadata remain unchanged.
        Metric UIDs are assigned by the host and preserved on updates.
        Every object must include aggregate SQL in ``expression``. ``derive``
        and ``window`` do not auto-generate it: follow the authoring skill's
        complete examples. Include dependencies first or in this same batch;
        if validation rejects a batch, none of its members are written.

        Args:
            path: Existing OSI semantic-model YAML file under the project workspace.
            metrics_json: Non-empty array of complete OSI metric objects. Prefer a
                structured array; a JSON-encoded array string is accepted for legacy callers.
        """
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error

        try:
            incoming_metrics = (
                json.loads(metrics_json) if isinstance(metrics_json, str) else copy.deepcopy(metrics_json)
            )
        except (TypeError, json.JSONDecodeError) as exc:
            return FuncToolResult(success=0, error=f"metrics_json is not a valid JSON-encoded array string: {exc}")
        if not isinstance(incoming_metrics, list) or not incoming_metrics:
            return FuncToolResult(success=0, error="metrics_json must be a non-empty JSON array")

        incoming_by_name: Dict[str, Dict[str, Any]] = {}
        for index, metric in enumerate(incoming_metrics):
            if not isinstance(metric, dict):
                return FuncToolResult(success=0, error=f"metrics_json[{index}] must be a JSON object")
            name = str(metric.get("name") or "").strip()
            if not name:
                return FuncToolResult(success=0, error=f"metrics_json[{index}].name is required")
            if name in incoming_by_name:
                return FuncToolResult(success=0, error=f"metrics_json contains duplicate metric name: {name}")
            incoming_by_name[name] = metric

        target_path = resolved.resolved
        if self.osi_target_state is None:
            return FuncToolResult(
                success=0,
                error="Bind an existing OSI semantic model before authoring metrics.",
                result={"code": "semantic_model_required"},
            )
        with semantic_artifact_lock(target_path):
            if self.osi_target_state is not None:
                try:
                    self._require_metric_revision(target_path)
                except ValueError as exc:
                    return FuncToolResult(
                        success=0,
                        error=str(exc),
                        result={"code": "semantic_model_target_invalid"},
                    )
            if not target_path.exists() or not target_path.is_file():
                return FuncToolResult(
                    success=0,
                    error=(
                        "The selected semantic model needs datasets before metrics can be authored. "
                        "Add or repair the datasets with upsert_osi_datasets before authoring metrics."
                    ),
                    result={"code": "semantic_model_required", "semantic_model_file": resolved.display},
                )

            try:
                original_content = target_path.read_bytes()
                document = yaml.safe_load(original_content.decode("utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                return FuncToolResult(success=0, error=f"Cannot load OSI semantic model {resolved.display}: {exc}")

            if not isinstance(document, dict):
                return FuncToolResult(success=0, error="OSI semantic model root must be a YAML object")
            models = document.get("semantic_model")
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
                return FuncToolResult(success=0, error="OSI document must contain exactly one semantic_model object")

            model = models[0]
            existing_metrics = model["metrics"] if "metrics" in model else []
            if not isinstance(existing_metrics, list) or any(
                not isinstance(metric, dict) for metric in existing_metrics
            ):
                return FuncToolResult(success=0, error="semantic_model[0].metrics must be a list of metric objects")

            metric_indexes: Dict[str, int] = {}
            for index, metric in enumerate(existing_metrics):
                name = str(metric.get("name") or "").strip()
                if name:
                    metric_indexes[name] = index

            created: List[str] = []
            updated: List[str] = []
            unchanged: List[str] = []
            try:
                for name, metric in incoming_by_name.items():
                    previous = existing_metrics[metric_indexes[name]] if name in metric_indexes else None
                    ensure_metric_uid(metric, previous if previous is not None else {})
                validate_metric_uids(
                    [
                        metric
                        for metric in existing_metrics
                        if str(metric.get("name") or "").strip() not in incoming_by_name
                    ]
                    + list(incoming_by_name.values())
                )
            except ValueError as exc:
                return FuncToolResult(success=0, error=f"Invalid OSI metric identity: {exc}")
            for name, metric in incoming_by_name.items():
                if name in metric_indexes:
                    index = metric_indexes[name]
                    if existing_metrics[index] == metric:
                        unchanged.append(name)
                    else:
                        existing_metrics[index] = metric
                        updated.append(name)
                else:
                    metric_indexes[name] = len(existing_metrics)
                    existing_metrics.append(metric)
                    created.append(name)

            if created or updated:
                model["metrics"] = existing_metrics
                validation_error = self._validate_osi_document(document)
                if validation_error:
                    return FuncToolResult(
                        success=0,
                        error=f"Invalid OSI metric update: {validation_error}",
                        result={"written": False, "semantic_model_file": resolved.display},
                    )

                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    if self.osi_target_state is not None:
                        self.osi_target_state.record_artifact_snapshot(target_path, original_content)
                    atomic_write_text(target_path, serialized)
                except OSError as exc:
                    return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
                self._notify_mutation(target_path)
                serialized_content = serialized.encode("utf-8")
            else:
                serialized_content = original_content
            if self.osi_target_state is not None:
                self.osi_target_state.record_metric_touch(
                    target_path,
                    serialized_content,
                    list(incoming_by_name),
                )

        return FuncToolResult(
            result={
                "message": f"Upserted {len(incoming_by_name)} OSI metric(s)",
                "semantic_model_file": resolved.display,
                "created": created,
                "updated": updated,
                "unchanged": unchanged,
            }
        )

    def delete_osi_metrics(self, path: str, metric_names: List[str]) -> FuncToolResult:
        """Delete named metrics from one bound OSI semantic-model file.

        Missing names are treated as already absent so interrupted or repeated
        cleanup can still be published to the Knowledge Base. This tool does
        not cascade to other metrics. Resolve live references first; the complete
        candidate must validate before deletion is written.

        Args:
            path: Bound OSI semantic-model YAML file under the project workspace.
            metric_names: Exact metric names to remove from the model.
        """
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error

        if not isinstance(metric_names, list):
            return FuncToolResult(success=0, error="metric_names must be a non-empty list of metric names")
        requested: List[str] = []
        requested_keys: set[str] = set()
        for value in metric_names:
            name = str(value or "").strip()
            normalized_name = normalize_metric_name(name)
            if normalized_name and normalized_name not in requested_keys:
                requested.append(name)
                requested_keys.add(normalized_name)
        if not requested:
            return FuncToolResult(success=0, error="metric_names must contain at least one non-empty metric name")

        target_path = resolved.resolved
        if self.osi_target_state is None:
            return FuncToolResult(
                success=0,
                error="Bind an existing OSI semantic model before deleting metrics.",
                result={"code": "semantic_model_required"},
            )

        with semantic_artifact_lock(target_path):
            try:
                self._require_metric_revision(target_path)
            except ValueError as exc:
                return FuncToolResult(
                    success=0,
                    error=str(exc),
                    result={"code": "semantic_model_target_invalid"},
                )
            if not target_path.exists() or not target_path.is_file():
                return FuncToolResult(
                    success=0,
                    error=(
                        "The selected semantic model must exist before metrics can be deleted. "
                        "Select or repair the target with plan_osi_semantic_model_target or "
                        "bind_osi_semantic_model_target before deleting metrics."
                    ),
                    result={"code": "semantic_model_required", "semantic_model_file": resolved.display},
                )

            try:
                original_content = target_path.read_bytes()
                document = yaml.safe_load(original_content.decode("utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                return FuncToolResult(success=0, error=f"Cannot load OSI semantic model {resolved.display}: {exc}")

            if not isinstance(document, dict):
                return FuncToolResult(success=0, error="OSI semantic model root must be a YAML object")
            models = document.get("semantic_model")
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
                return FuncToolResult(success=0, error="OSI document must contain exactly one semantic_model object")

            model = models[0]
            existing_metrics = model["metrics"] if "metrics" in model else []
            if not isinstance(existing_metrics, list) or any(
                not isinstance(metric, dict) for metric in existing_metrics
            ):
                return FuncToolResult(success=0, error="semantic_model[0].metrics must be a list of metric objects")

            requested_set = {normalize_metric_name(name) for name in requested}
            deleted = [
                str(metric.get("name") or "").strip()
                for metric in existing_metrics
                if normalize_metric_name(metric.get("name")) in requested_set
            ]
            deleted_set = {normalize_metric_name(name) for name in deleted}
            remaining_metrics = [
                metric for metric in existing_metrics if normalize_metric_name(metric.get("name")) not in requested_set
            ]
            already_absent = [name for name in requested if normalize_metric_name(name) not in deleted_set]

            if deleted:
                model["metrics"] = remaining_metrics
                validation_error = self._validate_osi_document(document)
                if validation_error:
                    return FuncToolResult(
                        success=0,
                        error=f"Invalid OSI metric deletion: {validation_error}",
                        result={"written": False, "semantic_model_file": resolved.display},
                    )
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    self.osi_target_state.record_artifact_snapshot(target_path, original_content)
                    atomic_write_text(target_path, serialized)
                except OSError as exc:
                    return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
                serialized_content = serialized.encode("utf-8")
            else:
                serialized_content = original_content

            # Even a byte-preserving retry invalidates old publication evidence:
            # the requested names may still exist as stale rows in the KB.
            self._notify_mutation(target_path)
            self.osi_target_state.record_metric_touch(target_path, serialized_content, requested)

        return FuncToolResult(
            result={
                "message": f"Processed deletion of {len(requested)} OSI metric(s)",
                "semantic_model_file": resolved.display,
                "requested": requested,
                "deleted": deleted,
                "already_absent": already_absent,
                "remaining": [
                    str(metric.get("name") or "").strip()
                    for metric in remaining_metrics
                    if str(metric.get("name") or "").strip()
                ],
            }
        )

    def rollback_failed_authoring(self) -> bool:
        """Restore the artifact revision captured before this request's first mutation."""
        state = self.osi_target_state
        if state is None or state.artifact_snapshot_content is None or not state.artifact_snapshot_path:
            return False

        target_path = Path(state.artifact_snapshot_path)
        original_content = state.artifact_snapshot_content
        with semantic_artifact_lock(target_path):
            try:
                if state.artifact_snapshot_existed:
                    restored_content = original_content.decode("utf-8")
                    atomic_write_text(target_path, restored_content)
                    rollback_content: Optional[bytes] = original_content
                else:
                    target_path.unlink(missing_ok=True)
                    rollback_content = None
            except (OSError, UnicodeDecodeError):
                return False
            if self.generation_evidence is not None:
                self.generation_evidence.record_artifact_mutation(target_path)
            state.record_artifact_rollback(rollback_content)
        return True

    def _validate_osi_document(self, document: Dict[str, Any] | str) -> Optional[str]:
        from datus.agent.node.semantic_authoring import (
            validate_osi_authoring_document,
        )

        return validate_osi_authoring_document(document)

    def _reject_write_policy(self, resolved: ResolvedPath) -> Optional[FuncToolResult]:
        if resolved.zone == PathZone.HIDDEN:
            return self._not_found(resolved)
        if self.strict and resolved.zone == PathZone.EXTERNAL:
            return self._strict_reject(resolved)
        if resolved.read_only:
            return self._read_only_reject(resolved)
        return None


class OsiSemanticModelFilesystemFuncTool(MetricFilesystemFuncTool):
    """OSI semantic-model filesystem surface with narrow dataset upserts."""

    def available_tools(self):
        """Expose model authoring plus a structure-preserving dataset mutation."""
        from datus.tools.func_tool import trans_to_function_tool

        return [
            trans_to_function_tool(self.read_file),
            trans_to_function_tool(self.read_image),
            trans_to_function_tool(self.edit_file),
            trans_to_function_tool(self.upsert_osi_datasets),
            trans_to_function_tool(self.delete_osi_datasets),
            trans_to_function_tool(self.glob),
            trans_to_function_tool(self.grep),
        ]

    def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:  # type: ignore[override]
        """Edit an existing OSI document only when the complete result is valid."""
        if not old_string:
            return FuncToolResult(success=0, error="old_string must not be empty")

        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error
        target_path = resolved.resolved

        with semantic_artifact_lock(target_path):
            guard_error = self._mutation_guard_error(target_path)
            if guard_error is not None:
                return guard_error
            if not target_path.exists():
                return FuncToolResult(success=0, error=f"File not found: {resolved.display}")
            if not target_path.is_file():
                return FuncToolResult(success=0, error=f"Path is not a file: {resolved.display}")
            try:
                content = target_path.read_text(encoding="utf-8")
                new_content, error = apply_single_replacement(content, old_string, new_string)
                if error is not None:
                    return FuncToolResult(success=0, error=error)
                document = yaml.safe_load(new_content)
            except UnicodeDecodeError:
                return FuncToolResult(success=0, error=f"Cannot edit binary file: {resolved.display}")
            except (OSError, yaml.YAMLError) as exc:
                return FuncToolResult(success=0, error=f"Cannot edit OSI semantic model {resolved.display}: {exc}")

            try:
                original_document = yaml.safe_load(content)
            except yaml.YAMLError:
                original_document = None

            if not isinstance(document, dict):
                return FuncToolResult(success=0, error="OSI semantic model root must be a YAML object")
            try:
                identities_changed = prepare_metric_uids(document, original_document or {})
            except ValueError as exc:
                return FuncToolResult(success=0, error=f"Invalid OSI metric identity: {exc}")
            # Validate the edit before any reserialization can hide duplicate keys.
            validation_error = self._validate_osi_document(new_content)
            if not validation_error and identities_changed:
                new_content = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                validation_error = self._validate_osi_document(new_content)
            if validation_error:
                return FuncToolResult(
                    success=0,
                    error=f"Invalid OSI semantic model edit: {validation_error}",
                    result={"written": False, "semantic_model_file": resolved.display},
                )
            try:
                if self.osi_target_state is not None:
                    self.osi_target_state.record_artifact_snapshot(target_path, content.encode("utf-8"))
                atomic_write_text(target_path, new_content)
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
            self._notify_mutation(target_path)
            if self.osi_target_state is not None:
                before_metrics = metric_definitions(original_document)
                after_metrics = metric_definitions(document)
                changed_metrics = [
                    (after_metrics.get(key) or before_metrics[key])["name"]
                    for key in sorted(before_metrics.keys() | after_metrics.keys())
                    if before_metrics.get(key) != after_metrics.get(key)
                ]
                if changed_metrics:
                    self.osi_target_state.record_metric_touch(target_path, new_content.encode("utf-8"), changed_metrics)
            if self.osi_target_state is not None and self.osi_target_state.planned is not None:
                before = self._dataset_definitions(original_document)
                after = self._dataset_definitions(document)
                changed = [
                    (after.get(key) or before[key])[0]
                    for key in before.keys() | after.keys()
                    if before.get(key) != after.get(key)
                ]
                if changed:
                    self.osi_target_state.record_planned_dataset_touch(changed)
        return FuncToolResult(result=f"File edited successfully: {resolved.display}")

    @staticmethod
    def _dataset_definitions(document: Any) -> Dict[str, tuple[str, Dict[str, Any]]]:
        """Index dataset definitions for planned edit change tracking."""
        models = document.get("semantic_model") if isinstance(document, dict) else None
        if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
            return {}
        return {
            str(dataset.get("name") or "").strip().casefold(): (str(dataset.get("name") or "").strip(), dataset)
            for dataset in models[0].get("datasets") or []
            if isinstance(dataset, dict) and str(dataset.get("name") or "").strip()
        }


class SemanticModelingFilesystemFuncTool(OsiSemanticModelFilesystemFuncTool):
    """Combined OSI surface for one dataset-and-metric authoring run."""

    def ensure_metric_uids(self, path: str) -> Dict[str, str]:
        """Host-only preparation before final validation; never an LLM tool."""
        target_path = self._classify(path).resolved
        with semantic_artifact_lock(target_path):
            self._require_metric_revision(target_path)
            original_content = target_path.read_bytes()
            document = yaml.safe_load(original_content)
            changed = prepare_metric_uids(document)
            if changed:
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                self.osi_target_state.record_artifact_snapshot(target_path, original_content)
                atomic_write_text(target_path, serialized)
                self._notify_mutation(target_path)
                self.osi_target_state.record_metric_touch(target_path, serialized.encode("utf-8"), changed)
            return {metric["name"]: metric_uid(metric) for metric in metric_definitions(document).values()}

    @staticmethod
    def _query_source_extensions(value: Any) -> List[Dict[str, Any]]:
        """Stamp Dosi query-backed datasets with the active extension version."""
        from datus.tools.semantic_tools.dosi.engine import datus_extension_version

        extensions = MetricFilesystemFuncTool._query_source_extensions(value)
        for extension in extensions:
            if not isinstance(extension, dict) or str(extension.get("vendor_name") or "").upper() != "DATUS":
                continue
            try:
                payload = json.loads(str(extension.get("data") or "{}"))
            except json.JSONDecodeError:
                payload = {}
            payload = payload if isinstance(payload, dict) else {}
            payload["v"] = datus_extension_version()
            payload["source_type"] = "query"
            extension["data"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            break
        return extensions

    def _require_metric_revision(self, path: str | Path) -> None:
        self.osi_target_state.require_selected_revision(path)

    def available_tools(self):
        """Expose the existing narrow dataset and metric mutation methods."""
        from datus.tools.func_tool import trans_to_function_tool

        return [
            trans_to_function_tool(self.read_file),
            trans_to_function_tool(self.read_image),
            trans_to_function_tool(self.edit_file),
            trans_to_function_tool(self.upsert_osi_datasets),
            trans_to_function_tool(self.delete_osi_datasets),
            trans_to_function_tool(self.upsert_osi_metrics),
            trans_to_function_tool(self.delete_osi_metrics),
            trans_to_function_tool(self.glob),
            trans_to_function_tool(self.grep),
        ]

    def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:  # type: ignore[override]
        """Replace one exact text occurrence on the selected target, validating before writing.

        Prefer dataset/metric upserts for those objects; use this for relationships or metadata.
        A failed whole-model validation writes nothing.

        Args:
            path: Path of the once-selected semantic model.
            old_string: Exact, nonempty text that occurs once in the current file.
            new_string: Replacement text for that occurrence.
        """
        return super().edit_file(path=path, old_string=old_string, new_string=new_string)
