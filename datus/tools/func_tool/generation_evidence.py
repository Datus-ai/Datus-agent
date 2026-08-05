# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Runtime evidence collected during generation workflows."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from datus.utils.exceptions import DatusException, ErrorCode


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
    validation, dry-run, and sync evidence.
    """

    validation_passed: bool = False
    metric_dry_run_passed: bool = False
    metric_dry_run_metrics: Set[str] = field(default_factory=set)
    metric_dry_run_queries: List[Dict[str, Any]] = field(default_factory=list)
    metric_sqls: Dict[str, str] = field(default_factory=dict)
    metric_queryability_contracts: List[Dict[str, Any]] = field(default_factory=list)
    required_metric_output_ids: List[str] = field(default_factory=list)
    semantic_kb_sync_passed: bool = False
    metric_kb_sync_passed: bool = False
    metric_kb_sync_metrics: Set[str] = field(default_factory=set)
    generic_kb_sync_passed: bool = False
    validated_semantic_artifacts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    sql_modeling_preflight_attempted: bool = False
    sql_modeling_plan_fingerprint: str = ""
    mutated_artifact_paths: Set[str] = field(default_factory=set)

    def reset(self) -> None:
        """Clear evidence before reusing a node for another request."""
        self.invalidate_artifact_evidence()
        self.metric_queryability_contracts.clear()
        self.required_metric_output_ids.clear()
        self.sql_modeling_preflight_attempted = False
        self.sql_modeling_plan_fingerprint = ""
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

    @property
    def sql_modeling_plan_status(self) -> str:
        """Expose a derived status without maintaining a mutable state machine."""
        if self.sql_modeling_plan_fingerprint:
            return "ready"
        return "unresolved" if self.sql_modeling_preflight_attempted else "pending"

    def mark_sql_modeling_preflight_attempted(self) -> None:
        """Remember that this request entered SQL preflight."""
        self.sql_modeling_preflight_attempted = True

    def mark_sql_modeling_plan_ready(self, source_fingerprint: str) -> None:
        """Record the immutable source fingerprint for a completed preflight."""
        fingerprint = str(source_fingerprint or "").strip()
        if not fingerprint:
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message="A ready SQL modeling plan requires a source fingerprint.",
            )
        self.sql_modeling_preflight_attempted = True
        self.sql_modeling_plan_fingerprint = fingerprint

    def require_sql_modeling_plan(self) -> None:
        """Reject authoring publication before the shared preflight completes."""
        if self.sql_modeling_plan_fingerprint:
            return
        raise DatusException(
            ErrorCode.TOOL_INVALID_INPUT,
            message="prepare_sql_modeling_plan must complete before publishing generated semantic artifacts.",
        )

    def ensure_sql_modeling_plan_resolved(self) -> bool:
        """Reject a failed preflight and report whether one completed successfully."""
        if self.sql_modeling_preflight_attempted and not self.sql_modeling_plan_fingerprint:
            self.require_sql_modeling_plan()
        return bool(self.sql_modeling_plan_fingerprint)

    def invalidate_artifact_evidence(self) -> None:
        """Discard validation, dry-run, and sync evidence after a file mutation."""
        self.validation_passed = False
        self.metric_dry_run_passed = False
        self.metric_dry_run_metrics.clear()
        self.metric_dry_run_queries.clear()
        self.metric_sqls.clear()
        self.semantic_kb_sync_passed = False
        self.metric_kb_sync_passed = False
        self.metric_kb_sync_metrics.clear()
        self.generic_kb_sync_passed = False
        self.validated_semantic_artifacts.clear()

    def invalidate_plan_evidence(self) -> None:
        """Discard plan-derived and downstream evidence before accepting a revised plan."""
        self.invalidate_artifact_evidence()
        self.metric_queryability_contracts.clear()
        self.required_metric_output_ids.clear()

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
    ) -> bool:
        """Bind successful validation to one model and the exact file content."""
        model_name = str(semantic_model_name or "").strip()
        state = self._semantic_artifact_state(path)
        if not model_name or state is None:
            return False
        if expected_sha256 and state["sha256"] != expected_sha256:
            return False
        self.validated_semantic_artifacts[model_name] = state
        return True

    def semantic_artifact_validation_passed(self, semantic_model_name: str, path: str | Path) -> bool:
        """Return whether the current artifact bytes match recorded validation evidence."""
        model_name = str(semantic_model_name or "").strip()
        expected = self.validated_semantic_artifacts.get(model_name)
        current = self._semantic_artifact_state(path)
        return expected is not None and current is not None and expected == current

    def set_metric_queryability_contracts(
        self,
        contracts: Optional[Iterable[Dict[str, Any]]],
    ) -> None:
        self.metric_queryability_contracts = []
        for contract in contracts or []:
            if not isinstance(contract, dict):
                continue
            dimensions = [
                str(dimension).strip() for dimension in contract.get("dimensions") or [] if str(dimension).strip()
            ]
            if not dimensions:
                continue
            normalized_contract = {
                "contract_id": str(contract.get("contract_id") or "").strip(),
                "source_id": str(contract.get("source_id") or "").strip(),
                "metric_output_ids": _deduplicate_preserve_order(
                    [
                        str(output_id).strip()
                        for output_id in contract.get("metric_output_ids") or []
                        if str(output_id).strip()
                    ]
                ),
                "dimensions": _deduplicate_preserve_order(dimensions),
            }
            time_grain = _normalize_time_grain(contract.get("time_grain"))
            if time_grain:
                normalized_contract["time_grain"] = time_grain
            self.metric_queryability_contracts.append(normalized_contract)

    def set_required_metric_outputs(self, requirements: Optional[Iterable[Dict[str, Any]]]) -> None:
        """Record the request-local output identities that must be published."""
        output_ids: List[str] = []
        seen: Set[str] = set()
        for requirement in requirements or []:
            if not isinstance(requirement, dict):
                continue
            role = str(requirement.get("role") or "metric").strip().lower()
            status = str(requirement.get("status") or "").strip().lower()
            if role != "metric" or status in {"ignored", "skipped", "blocked"}:
                continue
            output_id = str(requirement.get("output_id") or "").strip()
            if not output_id or output_id in seen:
                continue
            seen.add(output_id)
            output_ids.append(output_id)
        self.required_metric_output_ids = output_ids

    def bind_metric_output_names(self, bindings: Optional[Iterable[Dict[str, Any]]]) -> None:
        """Rewrite queryability contracts from SQL aliases to final published metric names."""
        names_by_output_id: Dict[str, str] = {}
        for binding in bindings or []:
            if not isinstance(binding, dict):
                continue
            output_id = str(binding.get("output_id") or "").strip()
            metric_name = str(binding.get("metric_name") or "").strip()
            if output_id and metric_name:
                names_by_output_id[output_id] = metric_name

        for contract in self.metric_queryability_contracts:
            output_ids = [
                str(output_id).strip()
                for output_id in contract.get("metric_output_ids") or []
                if str(output_id).strip()
            ]
            if not output_ids or any(output_id not in names_by_output_id for output_id in output_ids):
                continue
            final_names = _deduplicate_preserve_order([names_by_output_id[output_id] for output_id in output_ids])
            contract.setdefault("source_metric_hints", list(contract.get("metric_hints") or []))
            contract["metric_hints"] = final_names
            contract["metric_output_bindings"] = {output_id: names_by_output_id[output_id] for output_id in output_ids}

    def record_metric_dry_run(
        self,
        metrics: Optional[Iterable[str]],
        result: Any,
        dimensions: Optional[Iterable[str]] = None,
        time_granularity: Optional[str] = None,
    ) -> None:
        if not _result_success(result):
            return
        self.metric_dry_run_passed = True

        metric_candidates = [metrics] if isinstance(metrics, str) else list(metrics or [])
        dimension_candidates = [dimensions] if isinstance(dimensions, str) else list(dimensions or [])
        metrics_list = [m for m in metric_candidates if isinstance(m, str) and m]
        self.metric_dry_run_metrics.update(metrics_list)
        dimensions_list = [d for d in dimension_candidates if isinstance(d, str) and d]
        explicit_time_granularity = isinstance(time_granularity, str) and bool(_normalize_time_grain(time_granularity))
        normalized_time_granularity = (
            time_granularity if explicit_time_granularity else _time_grain_from_dimensions(dimensions_list)
        )
        dry_run_query = {
            "metrics": metrics_list,
            "dimensions": dimensions_list,
            "time_granularity": normalized_time_granularity,
            "time_granularity_explicit": explicit_time_granularity,
        }
        self.metric_dry_run_queries.append(dry_run_query)
        metadata = _metadata_from_result(result)
        metric_sqls = metadata.get("metric_sqls")
        if isinstance(metric_sqls, dict):
            combined_sql = metric_sqls.get("__query_metrics_dry_run__")
            if isinstance(combined_sql, str) and combined_sql.strip():
                dry_run_query["sql"] = combined_sql
            for name, sql in metric_sqls.items():
                if isinstance(name, str) and isinstance(sql, str) and sql:
                    self.metric_sqls[name] = sql
                    self.metric_dry_run_metrics.add(name)
            return

        sql = None
        for key in ("sql", "compiled_sql", "generated_sql", "dry_run_sql", "query"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                sql = value
                break
        if sql:
            dry_run_query["sql"] = sql
            if len(metrics_list) == 1:
                self.metric_sqls[metrics_list[0]] = sql
            else:
                self.metric_sqls["__query_metrics_dry_run__"] = sql

    def has_metric_dry_run(self, metric_names: Optional[Iterable[str]] = None) -> bool:
        names = {m for m in (metric_names or []) if isinstance(m, str) and m}
        if not names:
            return self.metric_dry_run_passed
        return self.metric_dry_run_passed and names.issubset(self.metric_dry_run_metrics)

    def has_required_queryability_dry_runs(self, metric_names: Optional[Iterable[str]] = None) -> bool:
        contracts = self.metric_queryability_contracts
        if not contracts:
            return True
        generated_metrics = {m for m in (metric_names or []) if isinstance(m, str) and m}
        for contract in contracts:
            if not self._contract_has_matching_dry_run(contract, generated_metrics):
                return False
        return True

    def missing_queryability_contracts(self, metric_names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        generated_metrics = {m for m in (metric_names or []) if isinstance(m, str) and m}
        return [
            contract
            for contract in self.metric_queryability_contracts
            if not self._contract_has_matching_dry_run(contract, generated_metrics)
        ]

    def _contract_has_matching_dry_run(self, contract: Dict[str, Any], generated_metrics: Set[str]) -> bool:
        metric_hints = {name for name in (contract.get("metric_hints") or []) if isinstance(name, str)}
        if generated_metrics:
            required_metrics = metric_hints & generated_metrics if metric_hints else generated_metrics
            if metric_hints and not required_metrics:
                return True
        else:
            required_metrics = metric_hints
        if not required_metrics and not metric_hints:
            required_metrics = generated_metrics

        for dry_run in self.metric_dry_run_queries:
            dry_run_metrics = {m for m in dry_run.get("metrics", []) if isinstance(m, str)}
            if required_metrics and not required_metrics.issubset(dry_run_metrics):
                continue
            if self._dimensions_satisfy_contract(dry_run, contract):
                return True
        return False

    def _dimensions_satisfy_contract(self, dry_run: Dict[str, Any], contract: Dict[str, Any]) -> bool:
        dimensions = {
            str(dimension).strip()
            for dimension in dry_run.get("dimensions", [])
            if isinstance(dimension, str) and dimension.strip()
        }
        required_dimensions = {
            str(dimension).strip()
            for dimension in contract.get("dimensions") or []
            if isinstance(dimension, str) and dimension.strip()
        }
        time_granularity = dry_run.get("time_granularity")
        required_time_grain = _normalize_time_grain(contract.get("time_grain"))
        if required_time_grain and _normalize_time_grain(time_granularity) != required_time_grain:
            return False
        return dimensions == required_dimensions

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


_TIME_GRAINS = {"day", "week", "month", "quarter", "year"}


def _deduplicate_preserve_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_time_grain(value: Any) -> str:
    text = str(value or "").strip().strip("'\"").lower()
    return text if text in _TIME_GRAINS else ""


def _time_grain_from_dimensions(dimensions: Iterable[str]) -> Optional[str]:
    for dimension in dimensions:
        grain = _dimension_time_grain(dimension)
        if grain:
            return grain
    return None


def _dimension_time_grain(dimension: str) -> str:
    text = str(dimension or "").strip().lower()
    if "__" not in text:
        return ""
    grain = re.sub(r"[^a-z0-9]+", "", text.rsplit("__", 1)[-1])
    return grain if grain in _TIME_GRAINS else ""
