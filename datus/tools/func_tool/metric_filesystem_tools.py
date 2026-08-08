# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import yaml

from datus.storage.metric.store import normalize_metric_name
from datus.storage.metric.subject_path import normalize_metric_subject_tree_tag
from datus.storage.semantic_model.artifact_file import atomic_write_text, semantic_artifact_lock
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.fs_path_policy import PathZone, ResolvedPath
from datus.utils.memory_loader import apply_single_replacement

if TYPE_CHECKING:
    from datus.tools.func_tool.generation_evidence import GenerationEvidence
    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState


class MetricFilesystemFuncTool(FilesystemFuncTool):
    """Filesystem tool variant for MetricFlow YAML generation.

    Batch metric generation often updates the same semantic model and metrics
    files across several batches. Plain ``write_file`` replacement can drop
    measures, dimensions, or metrics created by earlier batches, so existing
    MetricFlow YAML files are merged structurally.
    """

    def __init__(
        self,
        *args,
        authoring_format: str = "metricflow",
        semantic_adapter: str = "",
        osi_target_state: Optional["OsiSemanticModelTargetState"] = None,
        generation_evidence: Optional["GenerationEvidence"] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.authoring_format = (authoring_format or "metricflow").strip().lower()
        self.semantic_adapter = (
            (semantic_adapter or ("osi" if self.authoring_format == "osi" else "metricflow")).strip().lower()
        )
        self.osi_target_state = osi_target_state
        self.generation_evidence = generation_evidence

    def _is_metricflow_authoring(self) -> bool:
        return self.authoring_format == "metricflow"

    def available_tools(self):
        """Expose format-specific filesystem mutation tools.

        MetricFlow metric generation still needs the general write/edit surface
        because metrics can add measures and dimensions to semantic-model files.
        OSI metric generation uses narrow upsert/delete tools for metrics and
        datasets so it can repair required fields or add query-backed datasets
        without rewriting relationships or model metadata.
        """
        if self._is_metricflow_authoring():
            return super().available_tools()

        from datus.tools.func_tool import trans_to_function_tool

        return [
            trans_to_function_tool(self.read_file),
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

    def upsert_osi_datasets(self, path: str, datasets_json: str) -> FuncToolResult:
        """Create or update datasets in a planned OSI semantic-model file.

        The input is a JSON array of complete OSI dataset objects. Query-backed
        datasets may provide generated SQL directly in ``source``; the DATUS
        query-source extension is added automatically. A missing planned file
        is created with the first non-empty dataset batch.
        Existing datasets are replaced by ``name`` and new datasets are appended.
        Relationships, metrics, and model metadata are preserved.

        Args:
            path: Existing, planned OSI semantic-model YAML file.
            datasets_json: JSON array containing OSI dataset objects.
        """
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error
        target_path = resolved.resolved

        try:
            incoming_datasets = json.loads(datasets_json)
        except (TypeError, json.JSONDecodeError) as exc:
            return FuncToolResult(success=0, error=f"datasets_json must be a valid JSON array: {exc}")
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
                    return FuncToolResult(success=0, error=f"Invalid OSI dataset update: {validation_error}")
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    if self.osi_target_state is not None and self.osi_target_state.bound is not None:
                        self.osi_target_state.record_metric_snapshot(target_path, original_content)
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
        and model metadata are deliberately preserved so the authoring LLM can
        resolve any resulting semantic validation errors according to the
        user's intent instead of following hard-coded cascade rules. Missing
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
                    return FuncToolResult(success=0, error=f"Invalid OSI dataset deletion: {validation_error}")
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    if self.osi_target_state is not None and self.osi_target_state.bound is not None:
                        self.osi_target_state.record_metric_snapshot(target_path, original_content)
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

    def upsert_osi_metrics(self, path: str, metrics_json: str) -> FuncToolResult:
        """Create or update metrics in an existing OSI semantic-model file.

        The input is a JSON array of OSI metric objects. Existing metrics are
        replaced by ``name`` and new metrics are appended. Identical metrics
        leave the file bytes unchanged but still enter this request's exact
        publish scope. The tool only owns the ``metrics`` collection, so
        datasets, fields, relationships, and model metadata remain unchanged.

        Args:
            path: Existing OSI semantic-model YAML file under the project workspace.
            metrics_json: JSON array containing complete OSI metric objects.
        """
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error

        try:
            incoming_metrics = json.loads(metrics_json)
        except (TypeError, json.JSONDecodeError) as exc:
            return FuncToolResult(success=0, error=f"metrics_json must be a valid JSON array: {exc}")
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
        if not self._is_metricflow_authoring():
            if self.osi_target_state is None:
                return FuncToolResult(
                    success=0,
                    error="Bind an existing OSI semantic model before authoring metrics.",
                    result={"code": "semantic_model_required"},
                )
        with semantic_artifact_lock(target_path):
            if self.osi_target_state is not None:
                try:
                    self.osi_target_state.require_current_revision(target_path)
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
                        "OSI semantic model is required before metric generation. "
                        "Run gen_semantic_model first, then retry gen_metrics."
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
                    return FuncToolResult(success=0, error=f"Invalid OSI metric update: {validation_error}")

                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    if self.osi_target_state is not None:
                        self.osi_target_state.record_metric_snapshot(target_path, original_content)
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
        not infer dependencies or cascade to other metrics; run semantic
        validation after the edit and decide the next change from its result.

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
        if self._is_metricflow_authoring():
            return FuncToolResult(success=0, error="delete_osi_metrics is only available in OSI authoring mode")
        if self.osi_target_state is None:
            return FuncToolResult(
                success=0,
                error="Bind an existing OSI semantic model before deleting metrics.",
                result={"code": "semantic_model_required"},
            )

        with semantic_artifact_lock(target_path):
            try:
                self.osi_target_state.require_current_revision(target_path)
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
                        "OSI semantic model is required before deleting metrics. "
                        "Run gen_semantic_model first, then retry gen_metrics."
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
                    return FuncToolResult(success=0, error=f"Invalid OSI metric deletion: {validation_error}")
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                try:
                    self.osi_target_state.record_metric_snapshot(target_path, original_content)
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

    def rollback_failed_metric_authoring(self) -> bool:
        """Restore the artifact revision captured before this request authored metrics."""
        state = self.osi_target_state
        if state is None or state.metric_snapshot_content is None or not state.metric_snapshot_path:
            return False

        target_path = Path(state.metric_snapshot_path)
        original_content = state.metric_snapshot_content
        with semantic_artifact_lock(target_path):
            try:
                restored_content = original_content.decode("utf-8")
                atomic_write_text(target_path, restored_content)
            except (OSError, UnicodeDecodeError):
                return False
            self._notify_mutation(target_path)
            state.record_metric_rollback(original_content)
        return True

    def _validate_osi_document(self, document: Dict[str, Any]) -> Optional[str]:
        from datus.agent.node.semantic_authoring import (
            validate_osi_authoring_document,
        )

        return validate_osi_authoring_document(document, semantic_adapter=self.semantic_adapter)

    def write_file(self, path: str, content: str, file_type: str = "") -> FuncToolResult:  # type: ignore[override]
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error

        target_path = resolved.resolved
        preprocess_result = self._preprocess_yaml_content(target_path, content)
        if not preprocess_result.success:
            return preprocess_result
        content = str(preprocess_result.result or "")

        if not self._is_metricflow_authoring():
            return super().write_file(path, content, file_type)

        if self._is_metric_file_path(target_path):
            if not self._should_merge_metric_file(target_path):
                normalize_result = self._normalize_metric_subject_tree_tags(target_path, content)
                if not normalize_result.success:
                    return normalize_result
                content = str(normalize_result.result or "")
                return super().write_file(path, content, file_type)

            merge_result = self._merge_metric_content(target_path, content)
            if not merge_result.success:
                return merge_result
            result = super().write_file(path, str(merge_result.result or ""), file_type)
            if result.success:
                result.result = f"Metric file merged successfully: {resolved.display}"
            return result

        if not self._should_merge_semantic_model(target_path):
            return super().write_file(path, content, file_type)

        merge_result = self._merge_semantic_model_content(target_path, content)
        if not merge_result.success:
            return merge_result
        result = super().write_file(path, str(merge_result.result or ""), file_type)
        if result.success:
            result.result = f"Semantic model merged successfully: {resolved.display}"
        return result

    def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:  # type: ignore[override]
        resolved = self._classify(path)
        policy_error = self._reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error

        target_path = resolved.resolved
        original_content: Optional[str] = None
        should_restore = self._is_semantic_yaml_path(target_path) and target_path.exists() and target_path.is_file()
        if should_restore:
            try:
                original_content = target_path.read_text(encoding="utf-8")
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Cannot read YAML file before edit: {exc}")

        result = super().edit_file(path, old_string, new_string)
        if not result.success:
            return result
        if not self._is_semantic_yaml_path(target_path) or not target_path.exists():
            return result

        postprocess_result = self._postprocess_yaml_file(target_path)
        if not postprocess_result.success:
            if original_content is not None:
                try:
                    target_path.write_text(original_content, encoding="utf-8")
                except OSError as exc:
                    return FuncToolResult(
                        success=0,
                        error=f"{postprocess_result.error}; additionally failed to restore original file: {exc}",
                    )
            return postprocess_result
        return result

    def _reject_write_policy(self, resolved: ResolvedPath) -> Optional[FuncToolResult]:
        if resolved.zone == PathZone.HIDDEN:
            return self._not_found(resolved)
        if self.strict and resolved.zone == PathZone.EXTERNAL:
            return self._strict_reject(resolved)
        if resolved.read_only:
            return self._read_only_reject(resolved)
        return None

    def _postprocess_yaml_file(self, target_path: Path) -> FuncToolResult:
        try:
            content = target_path.read_text(encoding="utf-8")
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Cannot read edited YAML file: {exc}")

        preprocess_result = self._preprocess_yaml_content(target_path, content)
        if not preprocess_result.success:
            return preprocess_result
        normalized_content = str(preprocess_result.result or "")

        try:
            list(yaml.safe_load_all(normalized_content))
        except yaml.YAMLError as exc:
            return FuncToolResult(success=0, error=f"Cannot normalize invalid edited YAML file: {exc}")

        if self._is_metric_file_path(target_path):
            normalize_result = self._normalize_metric_subject_tree_tags(target_path, normalized_content)
            if not normalize_result.success:
                return normalize_result
            normalized_content = str(normalize_result.result or "")

        if normalized_content != content:
            try:
                target_path.write_text(normalized_content, encoding="utf-8")
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Cannot normalize edited YAML file: {exc}")
        return FuncToolResult(result=normalized_content)

    def _should_merge_semantic_model(self, target_path: Path) -> bool:
        if not target_path.exists() or not target_path.is_file():
            return False
        if target_path.suffix.lower() not in {".yml", ".yaml"}:
            return False
        parts = target_path.parts
        if "subject" not in parts or "semantic_models" not in parts:
            return False
        subject_idx = parts.index("subject")
        if len(parts) <= subject_idx + 1 or parts[subject_idx + 1] != "semantic_models":
            return False
        return "metrics" not in parts[subject_idx + 2 : -1]

    def _should_merge_metric_file(self, target_path: Path) -> bool:
        return target_path.exists() and target_path.is_file() and self._is_metric_file_path(target_path)

    def _is_metric_file_path(self, target_path: Path) -> bool:
        if target_path.suffix.lower() not in {".yml", ".yaml"}:
            return False
        parts = target_path.parts
        if "subject" not in parts or "semantic_models" not in parts:
            return False
        subject_idx = parts.index("subject")
        if len(parts) <= subject_idx + 1 or parts[subject_idx + 1] != "semantic_models":
            return False
        return "metrics" in parts[subject_idx + 2 : -1]

    def _is_semantic_yaml_path(self, target_path: Path) -> bool:
        if target_path.suffix.lower() not in {".yml", ".yaml"}:
            return False
        parts = target_path.parts
        if "subject" not in parts or "semantic_models" not in parts:
            return False
        subject_idx = parts.index("subject")
        return len(parts) > subject_idx + 1 and parts[subject_idx + 1] == "semantic_models"

    def _preprocess_yaml_content(self, target_path: Path, content: str) -> FuncToolResult:
        if not self._is_semantic_yaml_path(target_path):
            return FuncToolResult(result=content)
        return FuncToolResult(result=self._repair_invalid_yaml_single_quote_escapes(content))

    @staticmethod
    def _repair_invalid_yaml_single_quote_escapes(content: str) -> str:
        repaired = content
        changed = False
        for _ in range(20):
            try:
                list(yaml.safe_load_all(repaired))
                return repaired if changed else content
            except yaml.YAMLError as exc:
                if "unknown escape character" not in str(exc) or "'" not in str(exc):
                    return content
                offset = MetricFilesystemFuncTool._yaml_error_offset(repaired, getattr(exc, "problem_mark", None))
                if offset is None or offset <= 0:
                    return content
                if repaired[offset] != "'" or repaired[offset - 1] != "\\":
                    return content
                repaired = repaired[: offset - 1] + repaired[offset:]
                changed = True
        return content

    @staticmethod
    def _yaml_error_offset(content: str, mark: object) -> Optional[int]:
        line = getattr(mark, "line", None)
        column = getattr(mark, "column", None)
        if not isinstance(line, int) or not isinstance(column, int) or line < 0 or column < 0:
            return None
        lines = content.splitlines(keepends=True)
        if line >= len(lines) or column >= len(lines[line]):
            return None
        return sum(len(item) for item in lines[:line]) + column

    def _merge_semantic_model_content(self, target_path: Path, incoming_content: str) -> FuncToolResult:
        try:
            existing_docs = list(yaml.safe_load_all(target_path.read_text(encoding="utf-8")))
            incoming_docs = list(yaml.safe_load_all(incoming_content))
        except yaml.YAMLError as exc:
            return FuncToolResult(success=0, error=f"Cannot merge invalid semantic model YAML: {exc}")
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Cannot read existing semantic model file: {exc}")

        existing_idx, existing_doc, existing_ds = self._find_data_source_doc(existing_docs)
        _, _, incoming_ds = self._find_data_source_doc(incoming_docs)
        if existing_ds is None:
            return FuncToolResult(
                success=0,
                error="Cannot merge existing semantic model YAML without a data_source document.",
            )
        if incoming_ds is None:
            return FuncToolResult(
                success=0,
                error="Cannot merge semantic model YAML update without a data_source document.",
            )

        merged_ds, error = self._merge_data_sources(existing_ds, incoming_ds)
        if error:
            return FuncToolResult(success=0, error=error)

        merged_doc = dict(existing_doc or {})
        merged_doc["data_source"] = merged_ds
        existing_docs[existing_idx] = merged_doc
        merged_content = yaml.safe_dump_all(existing_docs, allow_unicode=True, sort_keys=False)
        return self._normalize_metric_subject_tree_tags(target_path, merged_content)

    def _merge_metric_content(self, target_path: Path, incoming_content: str) -> FuncToolResult:
        try:
            existing_docs = list(yaml.safe_load_all(target_path.read_text(encoding="utf-8")))
            incoming_docs = list(yaml.safe_load_all(incoming_content))
        except yaml.YAMLError as exc:
            return FuncToolResult(success=0, error=f"Cannot merge invalid metric YAML: {exc}")
        except OSError as exc:
            return FuncToolResult(success=0, error=f"Cannot read existing metric file: {exc}")

        existing_by_name: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        for idx, doc in enumerate(existing_docs):
            metric = self._metric_from_doc(doc)
            name = normalize_metric_name(metric.get("name") if metric else "")
            if name and metric is not None:
                existing_by_name[name] = (idx, metric)
        if not existing_by_name:
            return FuncToolResult(success=0, error="Cannot merge existing metric YAML without metric documents.")

        saw_incoming_metric = False
        for incoming_doc in incoming_docs:
            incoming_metric = self._metric_from_doc(incoming_doc)
            incoming_name = normalize_metric_name(incoming_metric.get("name") if incoming_metric else "")
            if not incoming_name or incoming_metric is None:
                continue
            saw_incoming_metric = True
            existing_entry = existing_by_name.get(incoming_name)
            if existing_entry is None:
                existing_by_name[incoming_name] = (len(existing_docs), incoming_metric)
                existing_docs.append(incoming_doc)
                continue

            existing_idx, existing_metric = existing_entry
            conflict_field = self._metric_definition_conflict(existing_metric, incoming_metric)
            if conflict_field:
                metric_name = incoming_metric.get("name") or existing_metric.get("name") or incoming_name
                return FuncToolResult(
                    success=0,
                    error=(
                        f"Refusing to overwrite metric '{metric_name}': field '{conflict_field}' differs. "
                        "Metric names must be unique within a datasource; preserve the existing definition "
                        "or choose a new metric name."
                    ),
                )
            merged_metric = self._merge_metric_fields(existing_metric, incoming_metric)
            merged_doc = dict(existing_docs[existing_idx] or {})
            merged_doc["metric"] = merged_metric
            existing_docs[existing_idx] = merged_doc
            existing_by_name[incoming_name] = (existing_idx, merged_metric)

        if not saw_incoming_metric:
            return FuncToolResult(success=0, error="Cannot merge metric YAML update without metric documents.")

        merged_content = yaml.safe_dump_all(existing_docs, allow_unicode=True, sort_keys=False)
        return self._normalize_metric_subject_tree_tags(target_path, merged_content)

    def _normalize_metric_subject_tree_tags(self, target_path: Path, content: str) -> FuncToolResult:
        try:
            docs = list(yaml.safe_load_all(content))
        except yaml.YAMLError as exc:
            return FuncToolResult(success=0, error=f"Cannot normalize invalid metric YAML: {exc}")

        datasource, table_name = self._metric_scope_from_path(target_path)
        changed = False
        for doc in docs:
            metric = self._metric_from_doc(doc)
            if metric is None:
                continue
            locked_metadata = metric.get("locked_metadata")
            if not isinstance(locked_metadata, dict):
                continue
            tags = locked_metadata.get("tags")
            if not isinstance(tags, list):
                continue
            normalized_tags = []
            for tag in tags:
                normalized = (
                    normalize_metric_subject_tree_tag(tag, datasource=datasource, table_name=table_name)
                    if isinstance(tag, str)
                    else tag
                )
                normalized_tags.append(normalized)
                changed = changed or normalized != tag
            locked_metadata["tags"] = normalized_tags

        if not changed:
            return FuncToolResult(result=content)
        return FuncToolResult(result=yaml.safe_dump_all(docs, allow_unicode=True, sort_keys=False))

    @staticmethod
    def _metric_scope_from_path(target_path: Path) -> Tuple[str, str]:
        parts = list(target_path.parts)
        datasource = ""
        if "semantic_models" in parts:
            idx = parts.index("semantic_models")
            if len(parts) > idx + 1:
                datasource = parts[idx + 1]
        stem = target_path.stem
        table_name = stem[: -len("_metrics")] if stem.endswith("_metrics") else stem
        return datasource, table_name or "Unknown"

    @staticmethod
    def _metric_from_doc(doc: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(doc, dict):
            return None
        metric = doc.get("metric")
        return metric if isinstance(metric, dict) else None

    @staticmethod
    def _merge_metric_fields(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(existing)
        for field, value in incoming.items():
            if field not in merged or merged.get(field) in (None, "", []):
                merged[field] = value
        return merged

    @classmethod
    def _metric_definition_conflict(cls, existing: Dict[str, Any], incoming: Dict[str, Any]) -> str:
        for field in ("type", "type_params"):
            existing_value = existing.get(field)
            incoming_value = incoming.get(field)
            if existing_value in (None, "", []) or incoming_value in (None, "", []):
                continue
            if cls._stable_yaml_value(existing_value) != cls._stable_yaml_value(incoming_value):
                return field
        return ""

    @staticmethod
    def _stable_yaml_value(value: Any) -> str:
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=True)

    @staticmethod
    def _find_data_source_doc(docs: List[Any]) -> Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        for idx, doc in enumerate(docs):
            if isinstance(doc, dict) and isinstance(doc.get("data_source"), dict):
                return idx, doc, doc["data_source"]
        return -1, None, None

    def _merge_data_sources(
        self,
        existing_ds: Dict[str, Any],
        incoming_ds: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        existing_name = str(existing_ds.get("name") or "").strip()
        incoming_name = str(incoming_ds.get("name") or "").strip()
        if existing_name and incoming_name and existing_name != incoming_name:
            return {}, (
                f"Refusing to overwrite semantic model '{existing_name}' with data_source '{incoming_name}'. "
                "Use a separate file for a different data_source."
            )

        merged = dict(existing_ds)
        for field in ("name", "description"):
            if not merged.get(field) and incoming_ds.get(field):
                merged[field] = incoming_ds[field]

        for field in ("sql_table", "sql_query"):
            error = self._merge_stable_scalar(merged, incoming_ds, field, existing_name or incoming_name)
            if error:
                return {}, error

        for field, conflict_fields in (
            ("identifiers", ("type", "expr")),
            ("measures", ("agg", "expr", "filter", "agg_params", "non_additive_dimension")),
            ("dimensions", ("type", "expr", "type_params")),
        ):
            merged_items, error = self._merge_named_items(
                field,
                merged.get(field) or [],
                incoming_ds.get(field) or [],
                conflict_fields,
                existing_name or incoming_name,
            )
            if error:
                return {}, error
            if merged_items:
                merged[field] = merged_items

        for field, value in incoming_ds.items():
            if field in {"name", "description", "sql_table", "sql_query", "identifiers", "measures", "dimensions"}:
                continue
            if field not in merged:
                merged[field] = value

        return merged, ""

    @staticmethod
    def _merge_stable_scalar(
        merged: Dict[str, Any],
        incoming: Dict[str, Any],
        field: str,
        data_source_name: str,
    ) -> str:
        existing_value = merged.get(field)
        incoming_value = incoming.get(field)
        if not existing_value and incoming_value:
            merged[field] = incoming_value
            return ""
        if existing_value and incoming_value and existing_value != incoming_value:
            return (
                f"Refusing to change data_source '{data_source_name}' field '{field}' from "
                f"{existing_value!r} to {incoming_value!r}. Edit intentionally or write a new data_source file."
            )
        return ""

    def _merge_named_items(
        self,
        section: str,
        existing_items: List[Any],
        incoming_items: List[Any],
        conflict_fields: Tuple[str, ...],
        data_source_name: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        merged: List[Dict[str, Any]] = [dict(item) for item in existing_items if isinstance(item, dict)]
        index_by_name = {
            str(item.get("name") or "").strip(): idx
            for idx, item in enumerate(merged)
            if str(item.get("name") or "").strip()
        }

        for raw_item in incoming_items:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            existing_idx = index_by_name.get(name)
            if existing_idx is None:
                index_by_name[name] = len(merged)
                merged.append(item)
                continue

            existing = merged[existing_idx]
            conflict_field = self._named_item_conflict(existing, item, conflict_fields)
            if conflict_field:
                return [], (
                    f"Refusing to overwrite {section[:-1]} '{name}' in data_source '{data_source_name}': "
                    f"field '{conflict_field}' differs. Preserve the existing definition or choose a new name."
                )
            for field, value in item.items():
                if field not in existing or existing.get(field) in (None, "", []):
                    existing[field] = value

        return merged, ""

    @staticmethod
    def _named_item_conflict(existing: Dict[str, Any], incoming: Dict[str, Any], fields: Tuple[str, ...]) -> str:
        for field in fields:
            existing_value = existing.get(field)
            incoming_value = incoming.get(field)
            if existing_value in (None, "", []) or incoming_value in (None, "", []):
                continue
            if existing_value != incoming_value:
                return field
        return ""


class OsiSemanticModelFilesystemFuncTool(MetricFilesystemFuncTool):
    """OSI semantic-model filesystem surface with narrow dataset upserts."""

    def __init__(self, *args, **kwargs):
        kwargs["authoring_format"] = "osi"
        super().__init__(*args, **kwargs)

    def available_tools(self):
        """Expose model authoring plus a structure-preserving dataset mutation."""
        from datus.tools.func_tool import trans_to_function_tool

        return [
            trans_to_function_tool(self.read_file),
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

            if not isinstance(document, dict):
                return FuncToolResult(success=0, error="OSI semantic model root must be a YAML object")
            validation_error = self._validate_osi_document(document)
            if validation_error:
                return FuncToolResult(success=0, error=f"Invalid OSI semantic model edit: {validation_error}")
            try:
                atomic_write_text(target_path, new_content)
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
            self._notify_mutation(target_path)
        return FuncToolResult(result=f"File edited successfully: {resolved.display}")
