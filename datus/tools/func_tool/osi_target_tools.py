# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Filesystem-backed OSI semantic-model planning and metric target binding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.base import FuncToolResult

if TYPE_CHECKING:
    from datus.tools.func_tool.generation_evidence import GenerationEvidence


def _normalize_metric_name(value: Any) -> str:
    return str(value or "").strip().lower()


@dataclass
class OsiSemanticModelTargetState:
    """Request-local exact target shared by OSI tools."""

    selected: Optional[Dict[str, Any]] = None
    mode: str = ""
    artifact_sha256: str = ""
    touched_metric_names: List[str] = field(default_factory=list)
    touched_dataset_names: List[str] = field(default_factory=list)
    planned_dataset_names: List[str] = field(default_factory=list)
    target_mutated: bool = False
    last_error_code: str = ""
    artifact_snapshot_path: str = ""
    artifact_snapshot_content: Optional[bytes] = None
    artifact_snapshot_existed: bool = False

    def clear_target(self) -> None:
        self.selected = None
        self.mode = ""
        self.artifact_sha256 = ""
        self.target_mutated = False

    def reset(self) -> None:
        self.clear_target()
        self.touched_metric_names = []
        self.touched_dataset_names = []
        self.planned_dataset_names = []
        self.last_error_code = ""
        self.artifact_snapshot_path = ""
        self.artifact_snapshot_content = None
        self.artifact_snapshot_existed = False

    def _matches_selected_target(self, candidate: Dict[str, Any]) -> bool:
        if self.selected is None:
            return False
        return (
            str(candidate.get("absolute_path") or "") == str(self.selected.get("absolute_path") or "")
            and str(candidate.get("semantic_model_name") or "") == str(self.selected.get("semantic_model_name") or "")
            and str(candidate.get("artifact_sha256") or "") == self.artifact_sha256
        )

    def select(self, candidate: Dict[str, Any], *, mode: str) -> None:
        if (
            self.target_mutated or self.touched_metric_names or self.touched_dataset_names or self.planned_dataset_names
        ) and not self._matches_selected_target(candidate):
            raise ValueError("The OSI target cannot change after authoring started.")
        self.selected = dict(candidate)
        self.mode = mode
        self.artifact_sha256 = str(candidate.get("artifact_sha256") or "")
        self.last_error_code = ""

    @property
    def bound(self) -> Optional[Dict[str, Any]]:
        return self.selected if self.mode == "bound" else None

    @property
    def planned(self) -> Optional[Dict[str, Any]]:
        return self.selected if self.mode == "planned" else None

    @property
    def selected_path(self) -> Optional[str]:
        if self.selected is None:
            return None
        path = str(self.selected.get("absolute_path") or "").strip()
        return path or None

    def require_bound_path(self, path: str | Path) -> Dict[str, Any]:
        bound = self.bound
        if bound is None:
            raise ValueError(
                "Bind an existing OSI semantic model with bind_osi_semantic_model_target before authoring metrics."
            )
        requested = Path(path).expanduser().resolve(strict=False)
        selected = Path(str(bound["absolute_path"])).expanduser().resolve(strict=False)
        if requested != selected:
            raise ValueError(f"Metric authoring is bound to {bound['semantic_model_file']}; refusing target {path!s}.")
        return bound

    def require_selected_path(self, path: str | Path) -> Dict[str, Any]:
        """Require the exact target selected once for this authoring run."""
        selected = self.selected
        if selected is None:
            raise ValueError("Select an OSI semantic-model target before authoring semantic objects.")
        requested = Path(path).expanduser().resolve(strict=False)
        expected = Path(str(selected["absolute_path"])).expanduser().resolve(strict=False)
        if requested != expected:
            raise ValueError(
                f"Semantic authoring is limited to {selected['semantic_model_file']}; refusing target {path!s}."
            )
        return selected

    def require_planned_path(self, path: str | Path) -> Dict[str, Any]:
        planned = self.planned
        if planned is None:
            raise ValueError("Plan the OSI semantic-model target before writing or editing semantic YAML.")
        requested = Path(path).expanduser().resolve(strict=False)
        selected = Path(str(planned["absolute_path"])).expanduser().resolve(strict=False)
        if requested != selected:
            raise ValueError(
                f"Semantic-model authoring is planned for {planned['semantic_model_file']}; refusing target {path!s}."
            )
        if self.artifact_sha256:
            try:
                current_sha256 = hashlib.sha256(requested.read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError(f"Cannot read the planned OSI semantic model: {exc}") from exc
            if current_sha256 != self.artifact_sha256:
                raise ValueError(
                    "The planned OSI semantic model changed after planning. Plan the target again before writing."
                )
        elif requested.exists():
            raise ValueError(
                "The planned OSI semantic-model path was created after planning. Plan the target again before writing."
            )
        return planned

    def require_current_revision(self, path: str | Path) -> None:
        self.require_bound_path(path)
        self._require_revision(path)

    def require_selected_revision(self, path: str | Path) -> None:
        """Require the current bytes of the once-selected target."""
        self.require_selected_path(path)
        self._require_revision(path)

    def _require_revision(self, path: str | Path) -> None:
        try:
            current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"Cannot read the bound OSI semantic model: {exc}") from exc
        if current != self.artifact_sha256:
            raise ValueError(
                "The bound OSI semantic model changed after selection. "
                "Inspect the live inventory and bind it again before writing."
            )

    def record_planned_write(self) -> None:
        planned = self.planned
        if planned is None:
            raise ValueError("Cannot record an OSI semantic-model write without a planned target.")
        path = Path(str(planned["absolute_path"]))
        self.artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.target_mutated = True

    def record_selected_write(self) -> None:
        """Refresh the revision after a write to the once-selected target."""
        selected = self.selected
        if selected is None:
            raise ValueError("Cannot record an OSI semantic-model write without a selected target.")
        path = Path(str(selected["absolute_path"]))
        self.artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.target_mutated = True

    def record_planned_dataset_touch(self, dataset_names: Iterable[str]) -> None:
        """Remember datasets changed during planned semantic-model authoring."""
        if self.planned is None:
            raise ValueError("Cannot record a changed dataset without a planned OSI target.")
        for name in dataset_names:
            normalized_name = str(name or "").strip().casefold()
            if normalized_name and not any(
                str(changed or "").strip().casefold() == normalized_name for changed in self.planned_dataset_names
            ):
                self.planned_dataset_names.append(str(name).strip())

    def record_metric_touch(self, path: str | Path, content: bytes, metric_names: List[str]) -> None:
        """Record explicit metric changes for final-YAML reconciliation."""
        self.require_selected_path(path)
        self.artifact_sha256 = hashlib.sha256(content).hexdigest()
        self._record_touched_metric_names(metric_names)

    def record_dataset_touch(self, path: str | Path, content: bytes, dataset_names: List[str]) -> None:
        """Record a narrow dataset change made against the selected target."""
        self.require_selected_path(path)
        self.artifact_sha256 = hashlib.sha256(content).hexdigest()
        self.target_mutated = True
        for name in dataset_names:
            normalized_name = str(name or "").strip().casefold()
            if normalized_name and not any(
                str(touched or "").strip().casefold() == normalized_name for touched in self.touched_dataset_names
            ):
                self.touched_dataset_names.append(str(name).strip())

    def partition_touched_metrics(self, current_metric_names: Iterable[str]) -> tuple[List[str], List[str]]:
        """Split touched names by presence in the final YAML, preserving canonical names."""
        current_names_by_key = {
            _normalize_metric_name(name): str(name).strip()
            for name in current_metric_names
            if _normalize_metric_name(name)
        }
        present: List[str] = []
        absent: List[str] = []
        for touched_name in self.touched_metric_names:
            current_name = current_names_by_key.get(_normalize_metric_name(touched_name))
            if current_name is None:
                absent.append(touched_name)
            else:
                present.append(current_name)
        return present, absent

    def _record_touched_metric_names(self, metric_names: List[str]) -> None:
        """Remember explicit upserts/deletes without encoding intermediate state."""
        for name in metric_names:
            normalized_name = _normalize_metric_name(name)
            if normalized_name and not any(
                _normalize_metric_name(touched) == normalized_name for touched in self.touched_metric_names
            ):
                self.touched_metric_names.append(str(name).strip())

    def record_artifact_snapshot(self, path: str | Path, content: bytes, *, existed: bool = True) -> None:
        """Keep the artifact revision from before this request's first mutation."""
        if self.artifact_snapshot_path:
            return
        self.require_selected_path(path)
        self.artifact_snapshot_path = str(Path(path).expanduser().resolve(strict=False))
        self.artifact_snapshot_content = bytes(content)
        self.artifact_snapshot_existed = bool(existed)

    def clear_artifact_snapshot(self) -> None:
        self.artifact_snapshot_path = ""
        self.artifact_snapshot_content = None
        self.artifact_snapshot_existed = False

    def record_artifact_rollback(self, content: Optional[bytes]) -> None:
        """Reset request-local mutation state after restoring the original artifact."""
        self.artifact_sha256 = hashlib.sha256(content).hexdigest() if content is not None else ""
        self.touched_metric_names = []
        self.touched_dataset_names = []
        self.planned_dataset_names = []
        self.target_mutated = False
        self.clear_artifact_snapshot()


class OsiSemanticModelTargetTools:
    """Expose one planner for model authoring and list/bind for metric authoring."""

    permission_category = "semantic_tools"

    def __init__(
        self,
        agent_config: AgentConfig,
        *,
        target_state: Optional[OsiSemanticModelTargetState] = None,
        generation_evidence: Optional["GenerationEvidence"] = None,
    ):
        self.agent_config = agent_config
        self.target_state = target_state or OsiSemanticModelTargetState()
        self.generation_evidence = generation_evidence
        self._inventory_cache: Optional[Dict[str, Any]] = None

    def available_tools(self):
        """Return the complete target-tool surface for permission registration."""
        from datus.tools.func_tool import trans_to_function_tool

        return [
            trans_to_function_tool(self.plan_osi_semantic_model_target),
            trans_to_function_tool(self.list_existing_osi_semantic_models),
            trans_to_function_tool(self.bind_osi_semantic_model_target),
        ]

    def _plan_failure(self, error: str, result: Optional[Dict[str, Any]] = None) -> FuncToolResult:
        if not self.target_state.target_mutated:
            self.target_state.reset()
        else:
            self.target_state.last_error_code = "semantic_model_target_invalid"
        return FuncToolResult(success=0, error=error, result=result)

    def _model_dir(self) -> Path:
        from datus.agent.node.semantic_authoring import osi_semantic_model_directory

        model_dir = osi_semantic_model_directory(self.agent_config)
        if model_dir is None:
            raise ValueError("The active datasource semantic-model directory is unavailable.")
        return model_dir.expanduser().resolve(strict=False)

    def _canonical_selector(self, semantic_model_file: str) -> Path:
        selector = str(semantic_model_file or "").strip()
        if not selector:
            raise ValueError("semantic_model_file is required")

        datasource = str(getattr(self.agent_config, "current_datasource", "") or "default").strip() or "default"
        model_dir = self._model_dir()
        expanded = Path(selector).expanduser()
        if expanded.is_absolute():
            candidate = expanded
        else:
            parts = PurePosixPath(selector.replace("\\", "/")).parts
            if parts[:2] == ("subject", "semantic_models"):
                if len(parts) < 4 or parts[2] != datasource:
                    raise ValueError(f"semantic_model_file must belong to active datasource {datasource!r}.")
                parts = parts[3:]
            candidate = model_dir.joinpath(*parts)

        canonical = candidate.resolve(strict=False)
        try:
            canonical.relative_to(model_dir)
        except ValueError as exc:
            raise ValueError(f"semantic_model_file must resolve inside the active datasource {datasource!r}.") from exc
        if canonical.suffix.lower() not in {".yml", ".yaml"}:
            raise ValueError("semantic_model_file must be a .yml or .yaml file")
        return canonical

    def invalidate_inventory(self) -> None:
        """Drop the request-local inventory snapshot."""
        self._inventory_cache = None

    @staticmethod
    def _public_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: candidate[key]
            for key in (
                "semantic_model_name",
                "semantic_model_file",
                "description",
                "datasets",
                "table_references",
                "relationships",
                "metrics",
                "repair_required",
            )
            if candidate.get(key)
        }

    @staticmethod
    def _public_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: issue[key]
            for key in (
                "code",
                "semantic_model_name",
                "semantic_model_file",
                "model_index",
                "dataset_name",
                "error",
            )
            if issue.get(key) is not None
        }

    def _inventory(self) -> Dict[str, Any]:
        from datus.agent.node.semantic_authoring import inspect_osi_semantic_model_inventory

        if self._inventory_cache is None:
            self._inventory_cache = inspect_osi_semantic_model_inventory(self.agent_config)
        return self._inventory_cache

    def _require_inventory_revision(self, candidate: Dict[str, Any]) -> None:
        """Ensure a cached candidate still refers to the same file bytes."""
        expected = str(candidate.get("artifact_sha256") or "")
        path = Path(str(candidate.get("absolute_path") or ""))
        try:
            current = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            self.invalidate_inventory()
            raise ValueError(f"The selected OSI semantic model is no longer readable: {exc}") from exc
        if not expected or current != expected:
            self.invalidate_inventory()
            raise ValueError(
                "The OSI semantic-model inventory changed after inspection. "
                "Call list_existing_osi_semantic_models again before selecting a target."
            )

    def plan_osi_semantic_model_target(
        self,
        semantic_model_name: str = "",
        business_domain: str = "",
        fact_tables: Optional[List[str]] = None,
        dimension_tables: Optional[List[str]] = None,
    ) -> FuncToolResult:
        """Plan the stable name and file used by gen_semantic_model."""
        if not self.target_state.target_mutated:
            self.target_state.reset()
        try:
            from datus.agent.node.semantic_authoring import (
                is_osi_authoring,
                plan_osi_semantic_model_target,
            )

            if not is_osi_authoring(self.agent_config):
                return self._plan_failure("OSI target planning is only available in OSI mode.")
            target = plan_osi_semantic_model_target(
                self.agent_config,
                semantic_model_name=semantic_model_name,
                business_domain=business_domain,
                fact_tables=fact_tables,
                dimension_tables=dimension_tables,
                inventory=self._inventory(),
            )
            if target.get("ambiguous"):
                return self._plan_failure(
                    target.get("reason") or "The OSI semantic-model target is ambiguous.",
                    target,
                )

            canonical = self._canonical_selector(str(target["semantic_model_file"]))
            selected = {
                **target,
                "absolute_path": str(canonical),
            }
            selected.setdefault("artifact_sha256", "")
            if target.get("exists"):
                self._require_inventory_revision(selected)
                target_path = Path(str(target.get("absolute_path") or "")).resolve(strict=False)
                if (
                    target_path != canonical
                    or not target.get("artifact_sha256")
                    or not target.get("semantic_model_name")
                ):
                    return self._plan_failure(
                        "The planned existing OSI target is not uniquely present in the live YAML inventory.",
                        {"code": "semantic_model_target_invalid"},
                    )
            if self.generation_evidence is not None:
                self.generation_evidence.invalidate_artifact_evidence()
            self.target_state.select(selected, mode="planned")
            result = self._public_candidate(selected) | {"exists": bool(target.get("exists"))}
            if selected.get("authoring_outline"):
                result["authoring_outline"] = selected["authoring_outline"]
            self.invalidate_inventory()
            return FuncToolResult(result=result)
        except Exception as exc:
            return self._plan_failure(f"Failed to plan OSI semantic-model target: {exc}")

    def list_existing_osi_semantic_models(self) -> FuncToolResult:
        """Return the live YAML inventory for LLM semantic target selection."""
        try:
            inventory = self._inventory()
            models = inventory["models"]
            repairable_models = inventory["recoverable_models"]
            issues = inventory["issues"]
            discovery_warnings = inventory["discovery_warnings"]
            if models and issues:
                status, code = "partial", None
            elif models:
                status, code = "found", None
            elif repairable_models:
                status, code = "repairable", None
            elif issues:
                status, code = "invalid", "semantic_model_target_invalid"
            else:
                status, code = "missing", "semantic_model_required"
            return FuncToolResult(
                result={
                    "status": status,
                    "code": code,
                    "count": len(models),
                    "files_scanned": inventory["files_scanned"],
                    "semantic_models": [self._public_candidate(model) for model in models],
                    "repairable_semantic_models": [self._public_candidate(model) for model in repairable_models],
                    "issues": [self._public_issue(issue) for issue in issues],
                    "discovery_warnings": [self._public_issue(warning) for warning in discovery_warnings],
                    "bound_target": (
                        self._public_candidate(self.target_state.bound) if self.target_state.bound is not None else None
                    ),
                }
            )
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Failed to list OSI semantic models: {exc}")

    def _bind_failure(self, code: str, error: str, **result: Any) -> FuncToolResult:
        if not self.target_state.touched_metric_names:
            self.target_state.clear_target()
        self.target_state.last_error_code = code
        return FuncToolResult(success=0, error=error, result={"code": code, **result})

    def bind_osi_semantic_model_target(
        self,
        semantic_model_file: str = "",
        semantic_model_name: str = "",
    ) -> FuncToolResult:
        """Validate and bind one exact existing semantic model selected by the LLM."""
        try:
            if not str(semantic_model_file or "").strip() and not str(semantic_model_name or "").strip():
                return self._bind_failure(
                    "semantic_model_selection_required",
                    "Provide semantic_model_file or semantic_model_name from the live inventory.",
                )

            requested_path: Optional[Path] = None
            if semantic_model_file:
                try:
                    requested_path = self._canonical_selector(semantic_model_file)
                except ValueError as exc:
                    return self._bind_failure("semantic_model_target_invalid", str(exc))

            inventory = self._inventory()
            matches = list(inventory["models"])
            if requested_path is not None:
                matches = [
                    model for model in matches if Path(model["absolute_path"]).resolve(strict=False) == requested_path
                ]
                path_issues = [
                    issue
                    for issue in inventory["issues"]
                    if Path(issue["absolute_path"]).resolve(strict=False) == requested_path
                ]
                if path_issues:
                    return self._bind_failure(
                        "semantic_model_target_invalid",
                        "The requested YAML file is not safe for metric authoring.",
                        issues=[self._public_issue(issue) for issue in path_issues],
                    )
            if semantic_model_name:
                requested_name = str(semantic_model_name).strip()
                matches = [model for model in matches if model["semantic_model_name"] == requested_name]

            if not matches:
                if requested_path is not None:
                    return self._bind_failure(
                        "semantic_model_target_invalid",
                        "The requested OSI semantic-model file is not present in the live YAML inventory.",
                    )
                if not inventory["models"] and not inventory["issues"]:
                    return self._bind_failure(
                        "semantic_model_required",
                        "No OSI semantic model exists for the active datasource.",
                    )
                return self._bind_failure(
                    "semantic_model_target_invalid",
                    "The requested OSI semantic model is not present in the live YAML inventory.",
                )
            if len(matches) != 1:
                return self._bind_failure(
                    "semantic_model_selection_required",
                    "The selector matches multiple OSI semantic models; provide both exact file and model name.",
                    candidates=[self._public_candidate(model) for model in matches],
                )

            selected = matches[0]
            self._require_inventory_revision(selected)
            if self.generation_evidence is not None:
                self.generation_evidence.invalidate_artifact_evidence()
            self.target_state.select(selected, mode="bound")
            result = {
                "status": "bound",
                **self._public_candidate(selected),
                "authoring_outline": selected.get("authoring_outline") or {},
            }
            self.invalidate_inventory()
            return FuncToolResult(result=result)
        except ValueError as exc:
            return self._bind_failure("semantic_model_target_invalid", str(exc))
        except Exception as exc:
            return self._bind_failure(
                "semantic_model_target_invalid",
                f"Failed to bind OSI semantic-model target: {exc}",
            )
