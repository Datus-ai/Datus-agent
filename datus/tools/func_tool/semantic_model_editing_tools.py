# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Object-level semantic editing through a shared authoring workspace."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, Dict, List

import yaml

from datus.storage.metric.store import normalize_metric_name
from datus.storage.semantic_model.artifact_file import atomic_write_text, semantic_artifact_lock
from datus.tools.func_tool.base import FuncToolResult, load_tool_json, tool_schema, trans_to_function_tool
from datus.tools.registry.contracts import tool_contract
from datus.tools.semantic_tools.metric_identity import (
    ensure_metric_uid,
    validate_metric_uids,
)

if TYPE_CHECKING:
    from datus.tools.func_tool.semantic_model_workspace import SemanticModelWorkspace


class SemanticModelEditingTools:
    """Mutate datasets and metrics on the authoring node's selected OSI model."""

    permission_category = "semantic_tools"

    def __init__(self, workspace: SemanticModelWorkspace):
        """Share path policy, target revisions and mutation tracking with raw file edits."""
        self.workspace = workspace

    def set_tool_context(self, context: Any) -> None:
        """Bind execution context shared with the file operations in this request."""
        self.workspace.set_tool_context(context)

    @classmethod
    def all_tools_name(cls) -> List[str]:
        """Return the object mutations supplied by this provider."""
        return [
            method.__name__
            for method in (
                cls.upsert_osi_datasets,
                cls.delete_osi_datasets,
                cls.upsert_osi_metrics,
                cls.delete_osi_metrics,
            )
        ]

    def available_tools(self):
        """Convert only semantic object mutations, without exposing file operations."""
        return [trans_to_function_tool(getattr(self, name)) for name in self.all_tools_name()]

    @tool_schema(strict_mode=False)
    @tool_contract(access="write", direct_call=False)
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
        resolved = self.workspace.resolve_path(path)
        policy_error = self.workspace.reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error
        target_path = resolved.resolved

        try:
            incoming_datasets = (
                load_tool_json(datasets_json) if isinstance(datasets_json, str) else copy.deepcopy(datasets_json)
            )
        except (TypeError, ValueError) as exc:
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
                dataset["custom_extensions"] = self.workspace.query_source_extensions(dataset.get("custom_extensions"))
            name = str(dataset.get("name") or "").strip()
            if not name:
                return FuncToolResult(success=0, error=f"datasets_json[{index}].name is required")
            if name in incoming_by_name:
                return FuncToolResult(success=0, error=f"datasets_json contains duplicate dataset name: {name}")
            incoming_by_name[name] = dataset

        with semantic_artifact_lock(target_path):
            guard_error = self.workspace.mutation_guard_error(target_path)
            if guard_error is not None:
                return guard_error
            creating = not target_path.exists()
            original_content = b""
            if creating:
                planned = self.workspace.target_state.planned if self.workspace.target_state is not None else None
                planned_name = str((planned or {}).get("semantic_model_name") or "").strip()
                if not planned_name:
                    return FuncToolResult(
                        success=0,
                        error="Plan the OSI semantic-model target before creating its first dataset.",
                        result={"code": "semantic_model_target_required"},
                    )
                document = {
                    "version": "0.2.0.dev0",
                    "name": planned_name,
                    "datasets": [],
                    "relationships": [],
                    "metrics": [],
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
            models = [document] if "datasets" in document else None
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
                return FuncToolResult(success=0, error="OSI document must contain one root model with datasets")

            model = models[0]
            existing_datasets = model["datasets"] if "datasets" in model else []
            if not isinstance(existing_datasets, list) or any(
                not isinstance(dataset, dict) for dataset in existing_datasets
            ):
                return FuncToolResult(success=0, error="datasets must be a list of dataset objects")

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
                validation_error = self.workspace.validate_document(document)
                if validation_error:
                    return FuncToolResult(
                        success=0,
                        error=f"Invalid OSI dataset update: {validation_error}",
                        result={"written": False, "semantic_model_file": resolved.display},
                    )
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    if self.workspace.target_state is not None:
                        self.workspace.target_state.record_artifact_snapshot(
                            target_path,
                            original_content,
                            existed=not creating,
                        )
                    atomic_write_text(target_path, serialized)
                except OSError as exc:
                    return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
                self.workspace.notify_mutation(target_path)
                serialized_content = serialized.encode("utf-8")
            else:
                serialized_content = original_content
            if self.workspace.target_state is not None and self.workspace.target_state.bound is not None:
                self.workspace.target_state.record_dataset_touch(
                    target_path,
                    serialized_content,
                    list(incoming_by_name),
                )
            elif self.workspace.target_state is not None and self.workspace.target_state.planned is not None:
                self.workspace.target_state.record_planned_dataset_touch([*created, *updated])

        return FuncToolResult(
            result={
                "message": f"Upserted {len(incoming_by_name)} OSI dataset(s)",
                "semantic_model_file": resolved.display,
                "created": created,
                "updated": updated,
                "unchanged": unchanged,
            }
        )

    @tool_contract(access="write", direct_call=False)
    def delete_osi_datasets(self, path: str, dataset_names: List[str]) -> FuncToolResult:
        """Delete explicitly named datasets from a planned OSI semantic model.

        Only ``datasets`` is changed. Relationships, metrics,
        and model metadata are preserved. Resolve live references first: the
        complete candidate must validate before deletion is written. Missing
        names are successful no-ops, allowing retries to republish the final
        YAML and clean stale Knowledge Base rows.

        Args:
            path: Existing, planned OSI semantic-model YAML file.
            dataset_names: Dataset business names to remove.
        """
        resolved = self.workspace.resolve_path(path)
        policy_error = self.workspace.reject_write_policy(resolved)
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
            guard_error = self.workspace.mutation_guard_error(target_path)
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
            models = [document] if "datasets" in document else None
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
                return FuncToolResult(success=0, error="OSI document must contain one root model with datasets")

            model = models[0]
            existing_datasets = model["datasets"] if "datasets" in model else []
            if not isinstance(existing_datasets, list) or any(
                not isinstance(dataset, dict) for dataset in existing_datasets
            ):
                return FuncToolResult(success=0, error="datasets must be a list of dataset objects")

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
                validation_error = self.workspace.validate_document(document)
                if validation_error:
                    return FuncToolResult(
                        success=0,
                        error=f"Invalid OSI dataset deletion: {validation_error}",
                        result={"written": False, "semantic_model_file": resolved.display},
                    )
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    if self.workspace.target_state is not None:
                        self.workspace.target_state.record_artifact_snapshot(target_path, original_content)
                    atomic_write_text(target_path, serialized)
                except OSError as exc:
                    return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
                serialized_content = serialized.encode("utf-8")
            else:
                serialized_content = original_content

            # A byte-preserving retry must still invalidate publication
            # evidence so stale semantic rows can be reconciled from the YAML.
            self.workspace.notify_mutation(target_path)
            if self.workspace.target_state is not None and self.workspace.target_state.bound is not None:
                self.workspace.target_state.record_dataset_touch(target_path, serialized_content, requested)

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

    @tool_schema(strict_mode=False)
    @tool_contract(access="write", direct_call=False)
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
        resolved = self.workspace.resolve_path(path)
        policy_error = self.workspace.reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error

        try:
            incoming_metrics = (
                load_tool_json(metrics_json) if isinstance(metrics_json, str) else copy.deepcopy(metrics_json)
            )
        except (TypeError, ValueError) as exc:
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
        if self.workspace.target_state is None:
            return FuncToolResult(
                success=0,
                error="Bind an existing OSI semantic model before authoring metrics.",
                result={"code": "semantic_model_required"},
            )
        with semantic_artifact_lock(target_path):
            if self.workspace.target_state is not None:
                try:
                    self.workspace.require_metric_revision(target_path)
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
            models = [document] if "datasets" in document else None
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
                return FuncToolResult(success=0, error="OSI document must contain one root model with datasets")

            model = models[0]
            existing_metrics = model["metrics"] if "metrics" in model else []
            if not isinstance(existing_metrics, list) or any(
                not isinstance(metric, dict) for metric in existing_metrics
            ):
                return FuncToolResult(success=0, error="metrics must be a list of metric objects")

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
                validation_error = self.workspace.validate_document(document)
                if validation_error:
                    return FuncToolResult(
                        success=0,
                        error=f"Invalid OSI metric update: {validation_error}",
                        result={"written": False, "semantic_model_file": resolved.display},
                    )

                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    if self.workspace.target_state is not None:
                        self.workspace.target_state.record_artifact_snapshot(target_path, original_content)
                    atomic_write_text(target_path, serialized)
                except OSError as exc:
                    return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
                self.workspace.notify_mutation(target_path)
                serialized_content = serialized.encode("utf-8")
            else:
                serialized_content = original_content
            if self.workspace.target_state is not None:
                self.workspace.target_state.record_metric_touch(
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

    @tool_contract(access="write", direct_call=False)
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
        resolved = self.workspace.resolve_path(path)
        policy_error = self.workspace.reject_write_policy(resolved)
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
        if self.workspace.target_state is None:
            return FuncToolResult(
                success=0,
                error="Bind an existing OSI semantic model before deleting metrics.",
                result={"code": "semantic_model_required"},
            )

        with semantic_artifact_lock(target_path):
            try:
                self.workspace.require_metric_revision(target_path)
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
            models = [document] if "datasets" in document else None
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
                return FuncToolResult(success=0, error="OSI document must contain one root model with datasets")

            model = models[0]
            existing_metrics = model["metrics"] if "metrics" in model else []
            if not isinstance(existing_metrics, list) or any(
                not isinstance(metric, dict) for metric in existing_metrics
            ):
                return FuncToolResult(success=0, error="metrics must be a list of metric objects")

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
                validation_error = self.workspace.validate_document(document)
                if validation_error:
                    return FuncToolResult(
                        success=0,
                        error=f"Invalid OSI metric deletion: {validation_error}",
                        result={"written": False, "semantic_model_file": resolved.display},
                    )
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    self.workspace.target_state.record_artifact_snapshot(target_path, original_content)
                    atomic_write_text(target_path, serialized)
                except OSError as exc:
                    return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
                serialized_content = serialized.encode("utf-8")
            else:
                serialized_content = original_content

            # Even a byte-preserving retry invalidates old publication evidence:
            # the requested names may still exist as stale rows in the KB.
            self.workspace.notify_mutation(target_path)
            self.workspace.target_state.record_metric_touch(target_path, serialized_content, requested)

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
