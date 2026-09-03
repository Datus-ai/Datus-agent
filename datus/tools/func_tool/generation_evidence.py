# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Runtime evidence collected during generation workflows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def _result_success(result: Any) -> bool:
    if isinstance(result, dict):
        return result.get("success") in (1, True)
    if hasattr(result, "success"):
        return result.success in (1, True)
    return False


def _result_payload(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("result")
    if hasattr(result, "result"):
        return result.result
    return None


def _metadata_from_result(result: Any) -> Dict[str, Any]:
    payload = _result_payload(result)
    if isinstance(payload, dict):
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            return metadata
    elif hasattr(payload, "metadata") and isinstance(payload.metadata, dict):
        return payload.metadata
    return {}


@dataclass
class GenerationEvidence:
    """Minimal runtime state for generation publish gates.

    Evidence is scoped to one node run. Exact semantic validation is tied to
    artifact bytes, and every successful authoring mutation invalidates prior
    validation, optional query SQL, and sync evidence.
    """

    validation_passed: bool = False
    metric_sqls: Dict[str, str] = field(default_factory=dict)
    semantic_kb_sync_passed: bool = False
    metric_kb_sync_passed: bool = False
    metric_kb_sync_metrics: Set[str] = field(default_factory=set)
    generic_kb_sync_passed: bool = False
    validated_semantic_artifacts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    compiled_metric_descriptions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compiled_metric_digests: Dict[str, str] = field(default_factory=dict)
    compiled_contract_digest: str = ""
    compiled_artifact_sha256: Dict[str, str] = field(default_factory=dict)
    compiled_metric_names: Set[str] = field(default_factory=set)
    mutated_artifact_paths: Set[str] = field(default_factory=set)

    def reset(self) -> None:
        """Clear evidence before reusing a node for another request."""
        self.invalidate_artifact_evidence()
        self.mutated_artifact_paths.clear()

    def record_artifact_mutation(self, path: str | Path | None = None) -> None:
        """Invalidate stale gates and remember the exact artifact that changed."""
        self.invalidate_artifact_evidence()
        if path is None:
            return
        try:
            normalized = str(Path(path).expanduser().resolve(strict=False))
        except (OSError, RuntimeError):
            normalized = str(path)
        if normalized:
            self.mutated_artifact_paths.add(normalized)

    def semantic_model_mutations(self, metric_file: str | Path = "") -> List[str]:
        """Return mutated YAML artifacts other than the metric collection."""
        metric_path = ""
        if metric_file:
            try:
                metric_path = str(Path(metric_file).expanduser().resolve(strict=False))
            except (OSError, RuntimeError):
                metric_path = str(metric_file)
        return sorted(
            path
            for path in self.mutated_artifact_paths
            if path != metric_path and "/metrics/" not in path.replace("\\", "/")
        )

    def invalidate_artifact_evidence(self) -> None:
        """Discard validation, optional query SQL, and sync evidence after a file mutation."""
        self.validation_passed = False
        self.metric_sqls.clear()
        self.semantic_kb_sync_passed = False
        self.metric_kb_sync_passed = False
        self.metric_kb_sync_metrics.clear()
        self.generic_kb_sync_passed = False
        self.validated_semantic_artifacts.clear()
        self.compiled_metric_descriptions.clear()
        self.compiled_metric_digests.clear()
        self.compiled_contract_digest = ""
        self.compiled_artifact_sha256.clear()
        self.compiled_metric_names.clear()

    @property
    def kb_sync_passed(self) -> bool:
        return self.semantic_kb_sync_passed or self.metric_kb_sync_passed or self.generic_kb_sync_passed

    def record_validation_result(self, result: Any) -> None:
        payload = _result_payload(result)
        valid = isinstance(payload, dict) and payload.get("valid") is True
        # Explicit adapter checks are diagnostic subsets. Only the adapter's
        # canonical default profile may satisfy a generation publish gate.
        canonical_profile = isinstance(payload, dict) and payload.get("checks") is None
        if _result_success(result) and valid and canonical_profile:
            self.validation_passed = True
            semantic_model_name = str(payload.get("semantic_model_name") or "").strip()
            semantic_model_file = str(payload.get("semantic_model_file") or "").strip()
            semantic_model_file_sha256 = str(payload.get("semantic_model_file_sha256") or "").strip()
            if semantic_model_name and semantic_model_file:
                self.record_semantic_artifact_validation(
                    semantic_model_name,
                    semantic_model_file,
                    expected_sha256=semantic_model_file_sha256,
                    validation_scope=str(payload.get("scope") or ""),
                )

    @staticmethod
    def _semantic_artifact_state(path: str | Path) -> Optional[Dict[str, str]]:
        try:
            resolved = Path(path).expanduser().resolve(strict=True)
            if not resolved.is_file():
                return None
            return {
                "path": str(resolved),
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            }
        except (OSError, RuntimeError):
            return None

    def record_semantic_artifact_validation(
        self,
        semantic_model_name: str,
        path: str | Path,
        *,
        expected_sha256: str = "",
        validation_scope: str = "",
    ) -> bool:
        """Bind successful validation to one model and the exact file content."""
        model_name = str(semantic_model_name or "").strip()
        state = self._semantic_artifact_state(path)
        if not model_name or state is None:
            return False
        if expected_sha256 and state["sha256"] != expected_sha256:
            return False
        existing = self.validated_semantic_artifacts.get(model_name)
        if (
            not validation_scope
            and existing is not None
            and all(existing.get(key) == state[key] for key in ("path", "sha256"))
        ):
            validation_scope = existing.get("scope", "")
        if validation_scope:
            state["scope"] = validation_scope
        self.validated_semantic_artifacts[model_name] = state
        return True

    def semantic_artifact_validation_passed(
        self,
        semantic_model_name: str,
        path: str | Path,
        *,
        required_scope: str = "",
    ) -> bool:
        """Return whether the current artifact bytes match recorded validation evidence."""
        model_name = str(semantic_model_name or "").strip()
        expected = self.validated_semantic_artifacts.get(model_name)
        current = self._semantic_artifact_state(path)
        if expected is None or current is None or any(expected.get(key) != current[key] for key in ("path", "sha256")):
            return False
        if not required_scope:
            return True
        validation_scope = str(expected.get("scope") or "")
        if required_scope == "semantic_model":
            return validation_scope in {"all", "semantic_model"}
        return validation_scope == required_scope

    @staticmethod
    def _normalize_sha256(value: Any) -> str:
        digest = str(value or "").strip().lower()
        if digest.startswith("sha256:"):
            digest = digest[7:]
        return digest if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) else ""

    def record_compiled_validation(
        self,
        metadata: Any,
        *,
        metric_names: Optional[Iterable[str]] = None,
    ) -> bool:
        """Record compiled metric semantics only when all identities are sound."""
        if not isinstance(metadata, dict):
            return False
        contract_digest = str(metadata.get("contract_digest") or "").strip()
        if not contract_digest.startswith("sha256:") or not self._normalize_sha256(contract_digest):
            return False

        artifacts = metadata.get("artifact_sha256")
        if not isinstance(artifacts, dict) or not artifacts:
            return False
        verified_artifacts: Dict[str, str] = {}
        for raw_path, raw_digest in artifacts.items():
            state = self._semantic_artifact_state(str(raw_path))
            digest = self._normalize_sha256(raw_digest)
            if state is None or not digest or state["sha256"] != digest:
                return False
            verified_artifacts[state["path"]] = digest

        descriptions: Dict[str, Dict[str, Any]] = {}
        for item in metadata.get("compiled_metrics") or []:
            if not isinstance(item, dict):
                return False
            name = str(item.get("name") or "").strip()
            if not name or name in descriptions:
                return False
            descriptions[name] = dict(item)

        raw_digests = metadata.get("compiled_metric_digests")
        if not isinstance(raw_digests, dict):
            return False
        digests = {
            str(name).strip(): str(digest).strip()
            for name, digest in raw_digests.items()
            if str(name).strip() and self._normalize_sha256(digest)
        }
        if set(digests) != set(descriptions):
            return False

        expected_names = (
            {str(name).strip() for name in metric_names if str(name).strip()}
            if metric_names is not None
            else set(descriptions)
        )
        if set(descriptions) != expected_names:
            return False

        self.compiled_metric_descriptions = descriptions
        self.compiled_metric_digests = digests
        self.compiled_contract_digest = contract_digest
        self.compiled_artifact_sha256 = verified_artifacts
        self.compiled_metric_names = expected_names
        return True

    def compiled_validation_passed(
        self,
        path: str | Path,
        metric_names: Iterable[str],
    ) -> bool:
        """Check artifact, contract, touched-set, and per-metric evidence identity."""
        state = self._semantic_artifact_state(path)
        expected_names = {str(name).strip() for name in metric_names if str(name).strip()}
        if state is None or self.compiled_artifact_sha256.get(state["path"]) != state["sha256"]:
            return False
        if not self.compiled_contract_digest or expected_names != self.compiled_metric_names:
            return False
        return expected_names == set(self.compiled_metric_descriptions) == set(self.compiled_metric_digests)

    def compiled_metric_catalog(self, metric_names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Return saved descriptions for an already identity-checked metric set."""
        names = {str(name).strip() for name in metric_names if str(name).strip()}
        if not names.issubset(self.compiled_metric_names):
            return {}
        return {name: dict(self.compiled_metric_descriptions[name]) for name in names}

    def record_metric_dry_run(
        self,
        metrics: Optional[Iterable[str]],
        result: Any,
    ) -> None:
        """Keep compiled metric SQL from an optional successful dry run."""
        if not _result_success(result):
            return

        metric_candidates = [metrics] if isinstance(metrics, str) else list(metrics or [])
        metrics_list = [m for m in metric_candidates if isinstance(m, str) and m]
        metadata = _metadata_from_result(result)
        metric_sqls = metadata.get("metric_sqls")
        if isinstance(metric_sqls, dict):
            for name, sql in metric_sqls.items():
                if isinstance(name, str) and isinstance(sql, str) and sql:
                    self.metric_sqls[name] = sql
            return

        sql = None
        for key in ("sql", "compiled_sql", "generated_sql", "dry_run_sql", "query"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                sql = value
                break
        if sql:
            if len(metrics_list) == 1:
                self.metric_sqls[metrics_list[0]] = sql
            else:
                self.metric_sqls["__query_metrics_dry_run__"] = sql

    def has_metric_kb_sync(self, metric_names: Optional[Iterable[str]] = None) -> bool:
        names = {str(name).strip() for name in (metric_names or []) if str(name).strip()}
        if not names:
            return False
        return self.metric_kb_sync_passed and names.issubset(self.metric_kb_sync_metrics)

    def mark_kb_sync(self, kind: str = "", metric_names: Optional[Iterable[str]] = None) -> None:
        if kind == "metric":
            self.metric_kb_sync_passed = True
            self.metric_kb_sync_metrics.update(str(name).strip() for name in (metric_names or []) if str(name).strip())
        elif kind == "semantic":
            self.semantic_kb_sync_passed = True
        else:
            self.generic_kb_sync_passed = True
