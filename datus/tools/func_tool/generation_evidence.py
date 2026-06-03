# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Runtime evidence collected during generation workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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

    The evidence is scoped to one node run and intentionally does not track
    file hashes or dirty state. The generation flow assumes files are not edited
    after successful validation / dry-run before publish.
    """

    validation_passed: bool = False
    metric_dry_run_passed: bool = False
    metric_dry_run_metrics: Set[str] = field(default_factory=set)
    metric_dry_run_queries: List[Dict[str, Any]] = field(default_factory=list)
    metric_sqls: Dict[str, str] = field(default_factory=dict)
    metric_queryability_contracts: List[Dict[str, Any]] = field(default_factory=list)
    semantic_kb_sync_passed: bool = False
    metric_kb_sync_passed: bool = False
    generic_kb_sync_passed: bool = False

    @property
    def kb_sync_passed(self) -> bool:
        return self.semantic_kb_sync_passed or self.metric_kb_sync_passed or self.generic_kb_sync_passed

    def record_validation_result(self, result: Any) -> None:
        payload = _result_payload(result)
        valid = isinstance(payload, dict) and payload.get("valid") is True
        if _result_success(result) and valid:
            self.validation_passed = True

    def set_metric_queryability_contracts(self, contracts: Optional[Iterable[Dict[str, Any]]]) -> None:
        self.metric_queryability_contracts = [
            contract for contract in (contracts or []) if isinstance(contract, dict) and contract.get("dimension_hints")
        ]

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
        self.metric_dry_run_queries.append(
            {
                "metrics": metrics_list,
                "dimensions": dimensions_list,
                "time_granularity": time_granularity if isinstance(time_granularity, str) else None,
            }
        )
        metadata = _metadata_from_result(result)
        metric_sqls = metadata.get("metric_sqls")
        if isinstance(metric_sqls, dict):
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
        required_metrics = {
            name
            for name in (contract.get("metric_hints") or [])
            if isinstance(name, str) and (not generated_metrics or name in generated_metrics)
        }
        if not required_metrics:
            required_metrics = generated_metrics

        for dry_run in self.metric_dry_run_queries:
            dry_run_metrics = {m for m in dry_run.get("metrics", []) if isinstance(m, str)}
            if required_metrics and not required_metrics.issubset(dry_run_metrics):
                continue
            if self._dimensions_satisfy_contract(dry_run, contract):
                return True
        return False

    def _dimensions_satisfy_contract(self, dry_run: Dict[str, Any], contract: Dict[str, Any]) -> bool:
        dimensions = [d for d in dry_run.get("dimensions", []) if isinstance(d, str)]
        time_granularity = dry_run.get("time_granularity")
        for hint in contract.get("dimension_hints") or []:
            if not isinstance(hint, str) or not hint.strip():
                continue
            if any(_dimension_matches_hint(dimension, hint) for dimension in dimensions):
                continue
            if (
                _looks_time_dimension(hint)
                and time_granularity
                and any(_is_metric_time_dimension(dimension) for dimension in dimensions)
            ):
                continue
            return False
        return True

    def mark_kb_sync(self, kind: str = "") -> None:
        if kind == "metric":
            self.metric_kb_sync_passed = True
        elif kind == "semantic":
            self.semantic_kb_sync_passed = True
        else:
            self.generic_kb_sync_passed = True


_GENERIC_DIMENSION_TOKENS = {"id", "key", "name", "dim", "dimension", "value"}
_TIME_DIMENSION_TOKENS = {"date", "time", "day", "week", "month", "quarter", "year", "ds"}


def _name_tokens(value: str) -> Set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(value).lower()) if token}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _semantic_tokens(value: str) -> Set[str]:
    tokens = _name_tokens(value)
    reduced = {token for token in tokens if token not in _GENERIC_DIMENSION_TOKENS}
    return reduced or tokens


def _looks_time_dimension(value: str) -> bool:
    return bool(_name_tokens(value) & _TIME_DIMENSION_TOKENS)


def _is_metric_time_dimension(value: str) -> bool:
    return str(value).strip().lower().startswith("metric_time")


def _dimension_matches_hint(dimension: str, hint: str) -> bool:
    normalized_dimension = _normalize_name(dimension)
    normalized_hint = _normalize_name(hint)
    if normalized_dimension == normalized_hint:
        return True
    dimension_tokens = _semantic_tokens(dimension)
    hint_tokens = _semantic_tokens(hint)
    if not dimension_tokens or not hint_tokens:
        return False
    if len(hint_tokens) > 1:
        return hint_tokens.issubset(dimension_tokens)
    return bool(dimension_tokens & hint_tokens)
