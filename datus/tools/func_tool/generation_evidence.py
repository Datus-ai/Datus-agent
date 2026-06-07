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
            contract
            for contract in (contracts or [])
            if isinstance(contract, dict) and (contract.get("dimension_hints") or contract.get("time_group_hints"))
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
        dry_run_query = {
            "metrics": metrics_list,
            "dimensions": dimensions_list,
            "time_granularity": time_granularity if isinstance(time_granularity, str) else None,
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
        required_metrics = {
            name
            for name in (contract.get("metric_hints") or [])
            if isinstance(name, str) and (not generated_metrics or name in generated_metrics)
        }
        if not required_metrics:
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
    dry_run_grain = _normalize_time_grain(dry_run.get("time_granularity"))
    if not grain or dry_run_grain != grain:
        return False

    dimensions = [d for d in dry_run.get("dimensions", []) if isinstance(d, str)]
    base_expr = time_hint.get("base_expr", "")
    if base_expr and any(_time_base_dimension_matches(dimension, base_expr) for dimension in dimensions):
        return True

    sql = dry_run.get("sql", "")
    if not isinstance(sql, str) or not _sql_contains_time_group(sql, base_expr, grain):
        return False
    return True


def _normalize_time_grain(value: Any) -> str:
    text = str(value or "").strip().strip("'\"").lower()
    return text if text in _TIME_GRAINS else ""


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
