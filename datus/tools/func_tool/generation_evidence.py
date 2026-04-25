# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Runtime evidence collected during generation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


def _result_success(result: Any) -> bool:
    if isinstance(result, dict):
        return result.get("success", 1) in (1, True)
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
    metric_sqls: Dict[str, str] = field(default_factory=dict)
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

    def record_metric_dry_run(self, metrics: Optional[Iterable[str]], result: Any) -> None:
        if not _result_success(result):
            return
        self.metric_dry_run_passed = True

        metrics_list = [m for m in (metrics or []) if isinstance(m, str) and m]
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
            for metric in metrics_list or ["__query_metrics_dry_run__"]:
                self.metric_sqls[metric] = sql

    def mark_kb_sync(self, kind: str = "") -> None:
        if kind == "metric":
            self.metric_kb_sync_passed = True
        elif kind == "semantic":
            self.semantic_kb_sync_passed = True
        else:
            self.generic_kb_sync_passed = True
