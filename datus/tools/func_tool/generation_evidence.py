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
    metric_aliases: Dict[str, str] = field(default_factory=dict)
    required_metric_output_ids: List[str] = field(default_factory=list)
    required_query_backed_sql: Dict[str, str] = field(default_factory=dict)
    query_backed_dataset_bindings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    semantic_kb_sync_passed: bool = False
    metric_kb_sync_passed: bool = False
    metric_kb_sync_metrics: Set[str] = field(default_factory=set)
    generic_kb_sync_passed: bool = False
    validated_semantic_artifacts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    sql_modeling_plan_status: str = "pending"
    sql_modeling_plan_fingerprint: str = ""
    mutated_artifact_paths: Set[str] = field(default_factory=set)

    def reset(self) -> None:
        """Clear evidence before reusing a node for another request."""
        self.invalidate_artifact_evidence()
        self.metric_queryability_contracts.clear()
        self.metric_aliases.clear()
        self.required_metric_output_ids.clear()
        self.required_query_backed_sql.clear()
        self.query_backed_dataset_bindings.clear()
        self.sql_modeling_plan_status = "pending"
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

    def set_sql_modeling_plan(self, status: str, source_fingerprint: str = "") -> None:
        """Record the request-local SQL preflight result."""
        normalized_status = str(status or "").strip().lower()
        if normalized_status not in {"ready", "unresolved"}:
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message=f"Unsupported SQL modeling plan status: {status!r}",
            )
        self.sql_modeling_plan_status = normalized_status
        self.sql_modeling_plan_fingerprint = str(source_fingerprint or "").strip()

    def require_sql_modeling_plan(self) -> None:
        """Reject authoring publication before the shared preflight completes."""
        if self.sql_modeling_plan_status == "ready":
            return
        raise DatusException(
            ErrorCode.TOOL_INVALID_INPUT,
            message="prepare_sql_modeling_plan must complete before publishing generated semantic artifacts.",
        )

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
        metric_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.metric_aliases = _normalized_metric_alias_map(metric_aliases or {})
        self.metric_queryability_contracts = []
        for contract in contracts or []:
            if not isinstance(contract, dict) or not (
                contract.get("dimension_hints") or contract.get("time_group_hints")
            ):
                continue
            normalized_contract = dict(contract)
            metric_hints = []
            alias_rewrites = {}
            for hint in contract.get("metric_hints") or []:
                if not isinstance(hint, str) or not hint.strip():
                    continue
                canonical = _canonical_metric_hint(hint, self.metric_aliases)
                metric_hints.append(canonical)
                if canonical != hint:
                    alias_rewrites[hint] = canonical
            if metric_hints:
                normalized_contract["metric_hints"] = _deduplicate_preserve_order(metric_hints)
            if alias_rewrites:
                normalized_contract["metric_alias_rewrites"] = alias_rewrites
            self.metric_queryability_contracts.append(normalized_contract)

    def set_required_metric_outputs(self, requirements: Optional[Iterable[Dict[str, Any]]]) -> None:
        """Record the request-local output identities that must be published."""
        output_ids: List[str] = []
        seen: Set[str] = set()
        for requirement in requirements or []:
            if not isinstance(requirement, dict):
                continue
            output_id = str(requirement.get("output_id") or "").strip()
            if not output_id or output_id in seen:
                continue
            seen.add(output_id)
            output_ids.append(output_id)
        self.required_metric_output_ids = output_ids

    def set_required_query_backed_datasets(self, requirements: Optional[Iterable[Dict[str, Any]]]) -> None:
        """Record exact SQL required to exist as authored query-backed datasets."""
        required: Dict[str, str] = {}
        for index, requirement in enumerate(requirements or [], 1):
            if not isinstance(requirement, dict):
                continue
            requirement_id = str(requirement.get("requirement_id") or f"query_dataset_{index}").strip()
            sql = str(requirement.get("sql") or "")
            if requirement_id and sql.strip():
                required[requirement_id] = sql
        self.required_query_backed_sql = required

    def query_backed_sql(self, requirement_id: str) -> str:
        """Resolve exact request-local SQL for one query-backed requirement."""
        return self.required_query_backed_sql.get(str(requirement_id or "").strip(), "")

    def query_backed_dataset_binding(self, requirement_id: str) -> Dict[str, str]:
        """Return the request-local dataset identity already chosen for a requirement."""
        return dict(
            self.query_backed_dataset_bindings.get(
                str(requirement_id or "").strip(),
                {},
            )
        )

    def bind_query_backed_dataset(
        self,
        requirement_id: str,
        *,
        semantic_model_file: str | Path,
        dataset_name: str,
    ) -> None:
        """Keep one query-backed requirement on one dataset throughout a request."""
        normalized_id = str(requirement_id or "").strip()
        normalized_name = str(dataset_name or "").strip()
        normalized_file = str(Path(semantic_model_file).expanduser().resolve(strict=False))
        if not normalized_id or not normalized_name:
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message="requirement_id and dataset_name are required",
            )

        candidate = {
            "semantic_model_file": normalized_file,
            "dataset_name": normalized_name,
        }
        existing = self.query_backed_dataset_bindings.get(normalized_id)
        if existing is not None and existing != candidate:
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message=(
                    f"Query-backed requirement {normalized_id!r} is already bound to dataset "
                    f"{existing['dataset_name']!r}."
                ),
            )
        self.query_backed_dataset_bindings[normalized_id] = candidate

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

        covered_metrics: Set[str] = set()
        for dry_run in self.metric_dry_run_queries:
            dry_run_metrics = {m for m in dry_run.get("metrics", []) if isinstance(m, str)}
            if required_metrics and not required_metrics.issubset(dry_run_metrics):
                if required_metrics and self._dimensions_satisfy_contract(dry_run, contract):
                    covered_metrics.update(required_metrics & dry_run_metrics)
                continue
            if self._dimensions_satisfy_contract(dry_run, contract):
                return True
        if required_metrics and required_metrics.issubset(covered_metrics):
            return True
        return False

    def _dimensions_satisfy_contract(self, dry_run: Dict[str, Any], contract: Dict[str, Any]) -> bool:
        dimensions = [d for d in dry_run.get("dimensions", []) if isinstance(d, str)]
        time_granularity = dry_run.get("time_granularity")
        for hint in contract.get("dimension_hints") or []:
            if not isinstance(hint, str) or not hint.strip():
                continue
            if _time_group_hint_satisfies(hint, dry_run, contract):
                continue
            if _has_time_group_hint_for_hint(hint, contract):
                return False
            if _dimension_expr_hint_satisfies(hint, dry_run, contract):
                continue
            if any(_dimension_matches_hint(dimension, hint) for dimension in dimensions):
                continue
            if (
                _looks_time_dimension(hint)
                and time_granularity
                and dry_run.get("time_granularity_explicit") is True
                and any(_is_metric_time_dimension(dimension) for dimension in dimensions)
            ):
                continue
            return False
        return True

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


_GENERIC_DIMENSION_TOKENS = {"id", "key", "name", "dim", "dimension", "value"}
_TIME_GRAINS = {"day", "week", "month", "quarter", "year"}
_TIME_DIMENSION_TOKENS = _TIME_GRAINS | {"date", "time", "ds"}
_SQL_PARSE_DIALECTS = (
    "snowflake",
    "bigquery",
    "duckdb",
    "mysql",
    "postgres",
    "postgresql",
    "sqlite",
    "starrocks",
    "trino",
    None,
)
_SQLGLOT_DIALECT_ALIASES = {"postgresql": "postgres"}


def _name_tokens(value: str) -> Set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(value).lower()) if token}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _deduplicate_preserve_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalized_metric_alias_map(metric_aliases: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for alias, canonical in metric_aliases.items():
        if not isinstance(alias, str) or not isinstance(canonical, str):
            continue
        alias = alias.strip()
        canonical = canonical.strip()
        if not alias or not canonical:
            continue
        normalized[alias] = canonical
        normalized[_normalize_name(alias)] = canonical
    return normalized


def _canonical_metric_hint(hint: str, metric_aliases: Dict[str, str]) -> str:
    return metric_aliases.get(hint) or metric_aliases.get(_normalize_name(hint)) or hint


def _semantic_tokens(value: str) -> Set[str]:
    tokens = _name_tokens(value)
    reduced = {token for token in tokens if token not in _GENERIC_DIMENSION_TOKENS}
    return reduced or tokens


def _looks_time_dimension(value: str) -> bool:
    return bool(_name_tokens(value) & _TIME_DIMENSION_TOKENS)


def _is_metric_time_dimension(value: str) -> bool:
    return str(value).strip().lower().startswith("metric_time")


def _time_group_hint_satisfies(hint: str, dry_run: Dict[str, Any], contract: Dict[str, Any]) -> bool:
    for time_hint in contract.get("time_group_hints") or []:
        if not isinstance(time_hint, dict):
            continue
        alias = time_hint.get("alias", "")
        base_expr = time_hint.get("base_expr", "")
        if not any(_dimension_matches_hint(candidate, hint) for candidate in (alias, base_expr) if candidate):
            continue
        if _dry_run_satisfies_time_group(dry_run, time_hint):
            return True
    return False


def _has_time_group_hint_for_hint(hint: str, contract: Dict[str, Any]) -> bool:
    for time_hint in contract.get("time_group_hints") or []:
        if not isinstance(time_hint, dict):
            continue
        alias = time_hint.get("alias", "")
        base_expr = time_hint.get("base_expr", "")
        if any(_dimension_matches_hint(candidate, hint) for candidate in (alias, base_expr) if candidate):
            return True
    return False


def _dimension_expr_hint_satisfies(hint: str, dry_run: Dict[str, Any], contract: Dict[str, Any]) -> bool:
    for expr_hint in contract.get("dimension_expr_hints") or []:
        if not isinstance(expr_hint, dict):
            continue
        alias = expr_hint.get("alias", "")
        expression = expr_hint.get("expr", "")
        if not _dimension_expr_hint_matches_hint(hint, alias, expression):
            continue
        if _dry_run_satisfies_dimension_expr(dry_run, expr_hint):
            return True
    return False


def _dimension_expr_hint_matches_hint(hint: str, alias: str, expression: str) -> bool:
    if alias and _dimension_matches_hint(alias, hint):
        return True
    if expression and _dimension_matches_hint(expression, hint):
        return True
    return False


def _dry_run_satisfies_dimension_expr(dry_run: Dict[str, Any], expr_hint: Dict[str, str]) -> bool:
    dimensions = [d for d in dry_run.get("dimensions", []) if isinstance(d, str)]
    if any(_dimension_matches_expr_hint(dimension, expr_hint) for dimension in dimensions):
        return True

    sql = dry_run.get("sql", "")
    expression = expr_hint.get("expr", "")
    return isinstance(sql, str) and _sql_contains_expression(sql, expression)


def _dimension_matches_expr_hint(dimension: str, expr_hint: Dict[str, str]) -> bool:
    normalized_dimension = _normalize_name(dimension)
    if not normalized_dimension:
        return False
    candidates = {
        _normalize_name(expr_hint.get("expr", "")),
        _normalize_name(expr_hint.get("column", "")),
    }
    return normalized_dimension in {candidate for candidate in candidates if candidate}


def _dry_run_satisfies_time_group(dry_run: Dict[str, Any], time_hint: Dict[str, str]) -> bool:
    grain = _normalize_time_grain(time_hint.get("grain", ""))
    dimensions = [d for d in dry_run.get("dimensions", []) if isinstance(d, str)]
    dry_run_grain = _normalize_time_grain(dry_run.get("time_granularity")) or _time_grain_from_dimensions(dimensions)
    if not grain or dry_run_grain != grain:
        return False

    base_expr = time_hint.get("base_expr", "")
    if base_expr and any(_time_base_dimension_matches(dimension, base_expr) for dimension in dimensions):
        return True

    sql = dry_run.get("sql", "")
    # MetricFlow canonicalizes the time dimension to metric_time, so accept it from
    # either the recorded dimensions or the compiled SQL.
    if (
        isinstance(sql, str)
        and base_expr
        and dimensions
        and _sql_contains_base_expr_text(sql, base_expr)
        and (any(_is_metric_time_dimension(dimension) for dimension in dimensions) or _sql_references_metric_time(sql))
    ):
        return True
    if not isinstance(sql, str) or not _sql_contains_time_group(sql, base_expr, grain):
        return False
    return True


def _sql_references_metric_time(sql: str) -> bool:
    """True when the compiled SQL references MetricFlow's metric_time column (any grain)."""
    return bool(re.search(r"\bmetric_time(?:__\w+)?\b", str(sql or ""), flags=re.IGNORECASE))


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


def _time_base_dimension_matches(dimension: str, base_expr: str) -> bool:
    if _dimension_matches_hint(dimension, base_expr):
        return True
    leaf_name = _last_identifier(base_expr)
    return bool(leaf_name and _dimension_matches_hint(dimension, leaf_name))


def _last_identifier(value: str) -> str:
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(value or ""))
    return identifiers[-1] if identifiers else ""


def _sql_contains_time_group(sql: str, base_expr: str, grain: str) -> bool:
    normalized_base = _normalize_sql_text(base_expr)
    if not normalized_base:
        return False
    for select in _parse_select_candidates(sql):
        for node in select.walk():
            expr = node[0] if isinstance(node, tuple) else node
            if _time_trunc_expression_matches(expr, normalized_base, grain):
                return True
    return False


def _sql_contains_base_expr_text(sql: str, base_expr: str) -> bool:
    normalized_sql = _normalize_sql_text(sql)
    normalized_base = _normalize_sql_text(base_expr)
    if normalized_base:
        # A bare identifier must match on identifier boundaries (against the
        # whitespace-preserving SQL, since _normalize_sql_text would fuse it with
        # the next token) so e.g. ``ordered_at`` matches a real column reference
        # but not ``preordered_at`` / ``ordered_at_utc``. Richer expressions
        # (containing parens/operators) are safe to match as a substring.
        if re.fullmatch(r"[a-z_][a-z0-9_]*", normalized_base):
            if re.search(rf"\b{re.escape(normalized_base)}\b", str(sql or "").lower()):
                return True
        elif normalized_base in normalized_sql:
            return True
    leaf = _last_identifier(base_expr)
    normalized_leaf = _normalize_sql_text(leaf)
    return bool(
        normalized_leaf and re.search(rf"(?<![a-z0-9_]){re.escape(normalized_leaf)}(?![a-z0-9_])", normalized_sql)
    )


def _sql_contains_expression(sql: str, expression: str) -> bool:
    normalized_expression = _normalize_sql_text(expression)
    if not normalized_expression:
        return False
    for select in _parse_select_candidates(sql):
        group = select.args.get("group")
        if group and any(_sql_expression_matches(expr, normalized_expression) for expr in group.expressions):
            return True
        for projection in select.expressions:
            expr = projection.this if projection.__class__.__name__ == "Alias" else projection
            if _sql_expression_matches(expr, normalized_expression):
                return True
    return False


def _parse_select_candidates(sql: str) -> Iterable[Any]:
    try:
        import sqlglot
        from sqlglot import expressions as exp

        seen = set()
        for dialect in _SQL_PARSE_DIALECTS:
            read_dialect = _SQLGLOT_DIALECT_ALIASES.get(dialect, dialect)
            try:
                parsed = sqlglot.parse_one(sql, read=read_dialect) if read_dialect else sqlglot.parse_one(sql)
            except Exception:
                continue
            select = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
            if select is None:
                continue
            key = _normalize_sql_text(select.sql(dialect="snowflake"))
            if key in seen:
                continue
            seen.add(key)
            yield select
    except Exception:
        return


def _time_trunc_expression_matches(expr: Any, normalized_base: str, grain: str) -> bool:
    try:
        from sqlglot import expressions as exp

        if not isinstance(expr, (exp.DateTrunc, exp.DatetimeTrunc, exp.TimeTrunc, exp.TimestampTrunc)):
            return False
        expr_grain = _normalize_time_grain(expr.args.get("unit"))
        if expr_grain != grain:
            return False
        base_expr = expr.args.get("this")
        return _sql_base_expression_matches(base_expr, normalized_base)
    except Exception:
        return False


def _sql_base_expression_matches(expr: Any, normalized_base: str) -> bool:
    normalized_expr = _normalize_sql_expression(expr)
    if normalized_expr == normalized_base:
        return True
    expr_leaf = _normalize_name(_last_identifier(normalized_expr))
    base_leaf = _normalize_name(_last_identifier(normalized_base))
    return bool(expr_leaf and base_leaf and expr_leaf == base_leaf)


def _sql_expression_matches(expr: Any, normalized_expression: str) -> bool:
    return _normalize_sql_expression(expr) == normalized_expression


def _normalize_sql_expression(expr: Any) -> str:
    if expr is None:
        return ""
    try:
        text = expr.sql(dialect="snowflake")
    except Exception:
        try:
            text = expr.sql()
        except Exception:
            text = str(expr)
    return _normalize_sql_text(text)


def _normalize_sql_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


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
