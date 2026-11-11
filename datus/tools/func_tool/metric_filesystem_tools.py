# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from datus.storage.semantic_model.artifact_file import atomic_write_text, semantic_artifact_lock
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.fs_path_policy import PathZone, ResolvedPath
from datus.tools.registry.contracts import tool_contract
from datus.tools.semantic_tools.metric_identity import (
    metric_definitions,
    metric_uid,
    prepare_metric_uids,
)
from datus.utils.memory_loader import apply_single_replacement

if TYPE_CHECKING:
    from datus.tools.func_tool.generation_evidence import GenerationEvidence
    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState


class MetricFilesystemFuncTool(FilesystemFuncTool):
    """File access and request-local guards shared with the semantic editor.

    The semantic editor owns object mutations; this surface owns raw file access.
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
        from datus.tools.func_tool.semantic_model_editing_tools import SemanticModelEditingTools

        self.semantic_editor = SemanticModelEditingTools(self)

    def _require_metric_revision(self, path: str | Path) -> None:
        self.osi_target_state.require_current_revision(path)

    def available_tools(self):
        """Expose the narrow OSI metric and dataset mutation tools."""
        from datus.tools.func_tool import trans_to_function_tool

        return [
            trans_to_function_tool(self.read_file),
            trans_to_function_tool(self.read_image),
            trans_to_function_tool(self.glob),
            trans_to_function_tool(self.grep),
        ]

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
            trans_to_function_tool(self.glob),
            trans_to_function_tool(self.grep),
        ]

    @tool_contract(access="write", direct_call=False)
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
        models = [document] if isinstance(document, dict) and "datasets" in document else None
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
            trans_to_function_tool(self.glob),
            trans_to_function_tool(self.grep),
        ]

    @tool_contract(access="write", direct_call=False)
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
