# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers for validating generated metrics against source-query shape."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_SQL_FENCE_LANGS = {
    "bigquery",
    "duckdb",
    "mysql",
    "postgres",
    "postgresql",
    "snowflake",
    "sql",
    "sqlite",
    "starrocks",
    "trino",
}
_FENCED_SQL_PATTERN = re.compile(r"```(?:\s*([^\n`]+))?\n(.*?)```", flags=re.IGNORECASE | re.DOTALL)


def extract_metric_queryability_contracts(text: Optional[str]) -> List[Dict[str, Any]]:
    """Extract queryability contracts from SQL evidence embedded in text.

    The contracts are intentionally adapter-neutral: they capture metric output
    aliases and GROUP BY dimension aliases from source SQL. Runtime validation
    later proves those dimensions are actually queryable through the configured
    semantic adapter.
    """
    contracts: List[Dict[str, Any]] = []
    for index, sql in enumerate(_extract_sql_snippets(text or "")):
        contract = _contract_from_sql(sql, source=f"sql_{index + 1}")
        if contract:
            contracts.append(contract)
    return contracts


def summarize_queryability_contracts(contracts: Iterable[Dict[str, Any]]) -> str:
    parts = []
    for contract in contracts:
        dimensions = ", ".join(contract.get("dimension_hints") or [])
        metrics = ", ".join(contract.get("metric_hints") or [])
        if dimensions:
            parts.append(f"{contract.get('source') or 'source SQL'} group-by [{dimensions}] metrics [{metrics}]")
    return "; ".join(parts)


def _extract_sql_snippets(text: str) -> List[str]:
    snippets: List[str] = []
    for match in _FENCED_SQL_PATTERN.finditer(text):
        fence_lang = (match.group(1) or "").strip().lower()
        candidate = match.group(2).strip()
        if fence_lang and fence_lang not in _SQL_FENCE_LANGS:
            continue
        if re.search(r"\bselect\b", candidate, flags=re.IGNORECASE):
            snippets.append(_strip_sql(candidate))

    remaining = _FENCED_SQL_PATTERN.sub(_fence_replacement_for_fallback, text)
    for match in re.finditer(r"(?is)\b(?:with\b.*?\bselect\b|select\b).*?(?:;|$)", remaining):
        candidate = _strip_sql(match.group(0))
        if candidate and candidate not in snippets:
            snippets.append(candidate)
    return snippets


def _fence_replacement_for_fallback(match: re.Match[str]) -> str:
    fence_lang = (match.group(1) or "").strip().lower()
    if not fence_lang or fence_lang in _SQL_FENCE_LANGS:
        return " "
    return f" {match.group(2)} "


def _strip_sql(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def _contract_from_sql(sql: str, source: str) -> Optional[Dict[str, Any]]:
    parsed = _parse_sql(sql)
    if parsed is None:
        return None

    select = _final_select(parsed)
    if select is None:
        return None

    dimension_hints: List[str] = []
    metric_hints: List[str] = []
    aliases_by_expr = _projection_aliases_by_expr(select)
    projection_aliases = [_projection_name(expr) for expr in select.expressions]

    group = select.args.get("group")
    if group:
        for group_expr in group.expressions:
            hint = _dimension_hint_from_group_expr(group_expr, aliases_by_expr, projection_aliases)
            if hint:
                dimension_hints.append(hint)

    grouped_hints = {_normalize_name(hint) for hint in dimension_hints}
    for projection in select.expressions:
        name = _projection_name(projection)
        if not name:
            continue
        if _normalize_name(name) in grouped_hints:
            continue
        if _looks_metric_projection(projection):
            metric_hints.append(name)

    dimension_hints = _dedupe(dimension_hints)
    metric_hints = _dedupe(metric_hints)
    if not dimension_hints:
        return None
    return {
        "source": source,
        "dimension_hints": dimension_hints,
        "metric_hints": metric_hints,
        "sql": sql,
    }


def _parse_sql(sql: str):
    try:
        import sqlglot

        for dialect in ("snowflake", None):
            try:
                return sqlglot.parse_one(sql, read=dialect) if dialect else sqlglot.parse_one(sql)
            except Exception:
                continue
    except Exception as exc:
        logger.debug(f"Skipping queryability SQL parsing: {exc}")
    return None


def _final_select(expression) -> Optional[Any]:
    try:
        from sqlglot import expressions as exp

        if isinstance(expression, exp.Select):
            return expression
        return expression.find(exp.Select)
    except Exception:
        return None


def _projection_aliases_by_expr(select) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for projection in select.expressions:
        name = _projection_name(projection)
        if not name:
            continue
        expr = projection.this if projection.__class__.__name__ == "Alias" else projection
        for key in {_normalize_sql(expr), _normalize_sql(projection)}:
            if key:
                aliases[key] = name
    return aliases


def _projection_name(projection) -> str:
    alias = getattr(projection, "alias", None)
    if alias:
        return str(alias)
    name = getattr(projection, "name", None)
    return str(name) if name else ""


def _dimension_hint_from_group_expr(group_expr, aliases_by_expr: Dict[str, str], projection_aliases: List[str]) -> str:
    try:
        from sqlglot import expressions as exp

        if isinstance(group_expr, exp.Literal) and group_expr.is_int:
            ordinal = int(group_expr.name)
            if 1 <= ordinal <= len(projection_aliases):
                return projection_aliases[ordinal - 1]
    except Exception:
        pass

    alias = aliases_by_expr.get(_normalize_sql(group_expr))
    if alias:
        return alias

    name = getattr(group_expr, "name", None)
    if name:
        return str(name)
    return _safe_name(getattr(group_expr, "sql", lambda: "")())


def _looks_metric_projection(projection) -> bool:
    try:
        from sqlglot import expressions as exp

        metric_classes = (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)
        expr = projection.this if isinstance(projection, exp.Alias) else projection
        return isinstance(expr, metric_classes) or any(
            isinstance(node[0] if isinstance(node, tuple) else node, metric_classes) for node in expr.walk()
        )
    except Exception:
        return False


def _normalize_sql(expr) -> str:
    if expr is None:
        return ""
    try:
        text = expr.sql(dialect="snowflake")
    except Exception:
        try:
            text = expr.sql()
        except Exception:
            text = str(expr)
    return re.sub(r"\s+", "", text).strip().lower()


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _safe_name(value: str) -> str:
    normalized = _normalize_name(value)
    return normalized[:80]


def _dedupe(values: Iterable[str]) -> List[str]:
    deduped = []
    seen = set()
    for value in values:
        normalized = _normalize_name(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped
