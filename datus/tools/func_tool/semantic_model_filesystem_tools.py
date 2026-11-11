# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Restricted file operations for OSI semantic-model authoring."""

from typing import Any, Dict, Optional

import yaml
from agents import Tool

from datus.storage.semantic_model.artifact_file import atomic_write_text, semantic_artifact_lock
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState
from datus.tools.func_tool.semantic_model_workspace import SemanticModelWorkspace
from datus.tools.registry.contracts import tool_contract
from datus.tools.semantic_tools.metric_identity import metric_definitions, prepare_metric_uids
from datus.utils.memory_loader import apply_single_replacement


class SemanticModelFilesystemTools(FilesystemFuncTool):
    """Reuse generic file access while validating and tracking raw model edits."""

    def __init__(
        self,
        *args,
        osi_target_state: Optional[OsiSemanticModelTargetState] = None,
        generation_evidence: Optional[GenerationEvidence] = None,
        **kwargs,
    ) -> None:
        """Create shared semantic state using the existing generic access policy."""
        super().__init__(*args, **kwargs)
        self.workspace = SemanticModelWorkspace(
            resolve_path=self._classify,
            reject_external_path=self._strict_reject,
            strict=self.strict,
            mutation_guard=self._mutation_guard_error,
            notify_mutation=self._notify_mutation,
            set_tool_context=self.set_tool_context,
            target_state=osi_target_state,
            generation_evidence=generation_evidence,
        )

    def available_tools(self) -> list[Tool]:
        """Expose read/search and validated text edits, with no raw write/delete."""
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
        if not old_string:
            return FuncToolResult(success=0, error="old_string must not be empty")

        resolved = self.workspace.resolve_path(path)
        policy_error = self.workspace.reject_write_policy(resolved)
        if policy_error is not None:
            return policy_error
        target_path = resolved.resolved

        with semantic_artifact_lock(target_path):
            guard_error = self.workspace.mutation_guard_error(target_path)
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
            validation_error = self.workspace.validate_document(new_content)
            if not validation_error and identities_changed:
                new_content = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                validation_error = self.workspace.validate_document(new_content)
            if validation_error:
                return FuncToolResult(
                    success=0,
                    error=f"Invalid OSI semantic model edit: {validation_error}",
                    result={"written": False, "semantic_model_file": resolved.display},
                )
            try:
                if self.workspace.target_state is not None:
                    self.workspace.target_state.record_artifact_snapshot(target_path, content.encode("utf-8"))
                atomic_write_text(target_path, new_content)
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Cannot update {resolved.display}: {exc}")
            self.workspace.notify_mutation(target_path)
            if self.workspace.target_state is not None:
                before_metrics = metric_definitions(original_document)
                after_metrics = metric_definitions(document)
                changed_metrics = [
                    (after_metrics.get(key) or before_metrics[key])["name"]
                    for key in sorted(before_metrics.keys() | after_metrics.keys())
                    if before_metrics.get(key) != after_metrics.get(key)
                ]
                if changed_metrics:
                    self.workspace.target_state.record_metric_touch(
                        target_path, new_content.encode("utf-8"), changed_metrics
                    )
            if self.workspace.target_state is not None and self.workspace.target_state.planned is not None:
                before = self._dataset_definitions(original_document)
                after = self._dataset_definitions(document)
                changed = [
                    (after.get(key) or before[key])[0]
                    for key in before.keys() | after.keys()
                    if before.get(key) != after.get(key)
                ]
                if changed:
                    self.workspace.target_state.record_planned_dataset_touch(changed)
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
