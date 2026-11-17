# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Request-local state and protections shared by semantic authoring tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from datus.storage.semantic_model.artifact_file import atomic_write_text, semantic_artifact_lock
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.fs_path_policy import PathZone, ResolvedPath
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState
from datus.tools.semantic_tools.metric_identity import metric_definitions, metric_uid, prepare_metric_uids


class SemanticModelWorkspace:
    """Share target revisions, validation and rollback across both editing surfaces.

    Access callbacks reuse the generic filesystem policy without depending on a
    semantic tool provider or maintaining another implementation of path rules.
    """

    def __init__(
        self,
        *,
        resolve_path: Callable[[str], ResolvedPath],
        reject_external_path: Callable[[ResolvedPath], FuncToolResult],
        strict: bool,
        mutation_guard: Callable[[Path], FuncToolResult | None],
        notify_mutation: Callable[[Path], None],
        set_tool_context: Callable[[Any], None],
        target_state: OsiSemanticModelTargetState | None = None,
        generation_evidence: GenerationEvidence | None = None,
    ) -> None:
        """Bind generic path access and request-local semantic authoring state."""
        self.resolve_path = resolve_path
        self._reject_external_path = reject_external_path
        self.strict = strict
        self.mutation_guard_error = mutation_guard
        self.notify_mutation = notify_mutation
        self.set_tool_context = set_tool_context
        self.target_state = target_state
        self.generation_evidence = generation_evidence

    def require_metric_revision(self, path: str | Path) -> None:
        """Require the current bytes of the selected existing or planned model."""
        self.target_state.require_selected_revision(path)

    def reject_write_policy(self, resolved: ResolvedPath) -> FuncToolResult | None:
        """Reject hidden, strict external and read-only paths before any write."""
        if resolved.zone == PathZone.HIDDEN:
            return FuncToolResult(success=0, error=f"File not found: {resolved.display}")
        if self.strict and resolved.zone == PathZone.EXTERNAL:
            return self._reject_external_path(resolved)
        if resolved.read_only:
            return FuncToolResult(success=0, error=f"Read-only path: {resolved.display} cannot be modified.")
        return None

    def rollback_failed_authoring(self) -> bool:
        """Restore the artifact revision captured before this request's first mutation."""
        state = self.target_state
        if state is None or state.artifact_snapshot_content is None or not state.artifact_snapshot_path:
            return False

        target_path = Path(state.artifact_snapshot_path)
        original_content = state.artifact_snapshot_content
        with semantic_artifact_lock(target_path):
            try:
                if state.artifact_snapshot_existed:
                    restored_content = original_content.decode("utf-8")
                    atomic_write_text(target_path, restored_content)
                    rollback_content: bytes | None = original_content
                else:
                    target_path.unlink(missing_ok=True)
                    rollback_content = None
            except (OSError, UnicodeDecodeError):
                return False
            if self.generation_evidence is not None:
                self.generation_evidence.record_artifact_mutation(target_path)
            state.record_artifact_rollback(rollback_content)
        return True

    def validate_document(self, document: dict[str, Any] | str) -> str | None:
        """Validate the complete OSI document before committing a mutation."""
        from datus.agent.node.semantic_authoring import (
            validate_osi_authoring_document,
        )

        return validate_osi_authoring_document(document)

    def ensure_metric_uids(self, path: str) -> dict[str, str]:
        """Host-only preparation before final validation; never an LLM tool."""
        target_path = self.resolve_path(path).resolved
        with semantic_artifact_lock(target_path):
            self.require_metric_revision(target_path)
            original_content = target_path.read_bytes()
            document = yaml.safe_load(original_content)
            changed = prepare_metric_uids(document)
            if changed:
                serialized = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
                self.target_state.record_artifact_snapshot(target_path, original_content)
                atomic_write_text(target_path, serialized)
                self.notify_mutation(target_path)
                self.target_state.record_metric_touch(target_path, serialized.encode("utf-8"), changed)
            return {metric["name"]: metric_uid(metric) for metric in metric_definitions(document).values()}

    @staticmethod
    def query_source_extensions(value: Any) -> list[dict[str, Any]]:
        """Stamp query-backed datasets with the active Datus extension version."""
        from datus.tools.semantic_tools.dosi.engine import datus_extension_version

        version = datus_extension_version()
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
                parsed["v"] = version
                extension["data"] = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            else:
                parsed = dict(data) if isinstance(data, dict) else {}
                parsed["source_type"] = "query"
                parsed["v"] = version
                extension["data"] = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            return extensions
        extensions.append(
            {
                "vendor_name": "DATUS",
                "data": json.dumps({"source_type": "query", "v": version}, sort_keys=True),
            }
        )
        return extensions
