# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers for validating generated metrics against source-query shape."""

from __future__ import annotations

import csv
import io
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
_FENCED_SQL_PATTERN = re.compile(r"```(?:\s*([^\n`]+))?\n(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_LABELED_SQL_PATTERN = re.compile(r"(?is)(?:^|\n)\s*SQL\s*:\s*(.*?)(?=\n\s*---\s*(?:\n|$)|\n\s*Query\s+\d+\s*:|$)")


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

    for candidate in _extract_labeled_sql_snippets(text):
        if candidate not in snippets:
            snippets.append(candidate)

    for candidate in _extract_csv_sql_snippets(text):
        if candidate not in snippets:
            snippets.append(candidate)

    remaining = _FENCED_SQL_PATTERN.sub(_fence_replacement_for_fallback, text)
    for match in re.finditer(r"(?is)\b(?:with\b.*?\bselect\b|select\b).*?(?:;|$)", remaining):
        candidate = _strip_sql(match.group(0))
        if candidate and candidate not in snippets:
            snippets.append(candidate)
    return snippets


def _extract_labeled_sql_snippets(text: str) -> List[str]:
    snippets: List[str] = []
    for match in _LABELED_SQL_PATTERN.finditer(text):
        candidate = _strip_sql(match.group(1))
        if re.search(r"\bselect\b", candidate, flags=re.IGNORECASE):
            snippets.append(candidate)
    return snippets


def _extract_csv_sql_snippets(text: str) -> List[str]:
    try:
        reader = csv.DictReader(io.StringIO(text))
    except csv.Error:
        return []
    if not reader.fieldnames or "sql" not in {str(name).strip().lower() for name in reader.fieldnames if name}:
        return []

    sql_field = next((name for name in reader.fieldnames if str(name).strip().lower() == "sql"), None)
    if not sql_field:
        return []

    snippets: List[str] = []
    try:
        for row in reader:
            candidate = _strip_sql(row.get(sql_field) or "")
            if re.search(r"\bselect\b", candidate, flags=re.IGNORECASE):
                snippets.append(candidate)
    except csv.Error:
        return []
    return snippets


def _fence_replacement_for_fallback(match: re.Match[str]) -> str:
    fence_lang = (match.group(1) or "").strip().lower()
    if not fence_lang or fence_lang in _SQL_FENCE_LANGS:
        return " "
    return f" {match.group(2)} "


def _strip_sql(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def _contract_from_sql(sql: str, source: str) -> Optional[Dict[str, Any]]:
    best_contract = None
    has_time_trunc = _contains_time_trunc(sql)
    for parsed in _parse_sql_candidates(sql):
        contract = _contract_from_parsed(parsed, sql, source)
        if contract is None:
            continue
        if contract.get("time_group_hints") or best_contract is None:
            best_contract = contract
        if contract.get("time_group_hints") or not has_time_trunc:
            break
    return best_contract


def _contract_from_parsed(parsed: Any, sql: str, source: str) -> Optional[Dict[str, Any]]:
    select = _final_select(parsed)
    if select is None:
        return None

    dimension_hints: List[str] = []
    metric_hints: List[str] = []
    dimension_expr_hints: List[Dict[str, str]] = []
    time_group_hints: List[Dict[str, str]] = []
    aliases_by_expr = _projection_aliases_by_expr(select)
    projection_aliases = [_projection_name(expr) for expr in select.expressions]

    group = select.args.get("group")
    if group:
        for group_expr in group.expressions:
            hint = _dimension_hint_from_group_expr(group_expr, aliases_by_expr, projection_aliases)
            if hint:
                dimension_hints.append(hint)
            time_group_hint = _time_group_hint_from_group_expr(group_expr, select.expressions)
            if time_group_hint:
                time_group_hints.append(time_group_hint)
            dimension_expr_hint = _dimension_expr_hint_from_group_expr(group_expr, select.expressions, hint)
            if dimension_expr_hint:
                dimension_expr_hints.append(dimension_expr_hint)

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
    contract = {
        "source": source,
        "dimension_hints": dimension_hints,
        "metric_hints": metric_hints,
        "sql": sql,
    }
    dimension_expr_hints = _dedupe_dimension_expr_hints(dimension_expr_hints)
    if dimension_expr_hints:
        contract["dimension_expr_hints"] = dimension_expr_hints
    time_group_hints = _dedupe_time_group_hints(time_group_hints)
    if time_group_hints:
        contract["time_group_hints"] = time_group_hints
    return contract


def _contains_time_trunc(sql: str) -> bool:
    return bool(re.search(r"\b(?:date|datetime|time|timestamp)_trunc\s*\(", sql, flags=re.IGNORECASE))


def _parse_sql_candidates(sql: str) -> Iterable[Any]:
    try:
        import sqlglot

        seen = set()
        for dialect in _SQL_PARSE_DIALECTS:
            read_dialect = _SQLGLOT_DIALECT_ALIASES.get(dialect, dialect)
            try:
                parsed = sqlglot.parse_one(sql, read=read_dialect) if read_dialect else sqlglot.parse_one(sql)
            except Exception:
                continue
            key = _normalize_sql(parsed)
            if key in seen:
                continue
            seen.add(key)
            yield parsed
    except Exception as exc:
        logger.debug(f"Skipping queryability SQL parsing: {exc}")


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


def _time_group_hint_from_group_expr(group_expr, projections: List[Any]) -> Optional[Dict[str, str]]:
    projection = _projection_for_group_expr(group_expr, projections)
    expr = projection.this if projection and projection.__class__.__name__ == "Alias" else projection
    if expr is None:
        expr = group_expr
    return _time_group_hint_from_expr(expr, alias=_projection_name(projection) if projection is not None else "")


def _projection_for_group_expr(group_expr, projections: List[Any]) -> Optional[Any]:
    try:
        from sqlglot import expressions as exp

        if isinstance(group_expr, exp.Literal) and group_expr.is_int:
            ordinal = int(group_expr.name)
            if 1 <= ordinal <= len(projections):
                return projections[ordinal - 1]
    except Exception:
        pass

    group_sql = _normalize_sql(group_expr)
    group_name = _normalize_name(getattr(group_expr, "name", ""))
    for projection in projections:
        alias = _normalize_name(_projection_name(projection))
        expr = projection.this if projection.__class__.__name__ == "Alias" else projection
        if alias and group_name and alias == group_name:
            return projection
        if group_sql and group_sql in {_normalize_sql(expr), _normalize_sql(projection)}:
            return projection
    return None


def _time_group_hint_from_expr(expr: Any, alias: str = "") -> Optional[Dict[str, str]]:
    try:
        if not _is_time_trunc_expr(expr):
            return None
        grain = _normalize_time_grain(expr.args.get("unit"))
        base_expr = expr.args.get("this")
        base = _sql_name(base_expr)
        if not grain or not base:
            return None
        hint = {
            "alias": alias or _safe_name(getattr(expr, "sql", lambda: "")()),
            "base_expr": base,
            "grain": grain,
        }
        return hint
    except Exception:
        return None


def _dimension_expr_hint_from_group_expr(group_expr, projections: List[Any], alias: str) -> Optional[Dict[str, str]]:
    if not alias:
        return None
    projection = _projection_for_group_expr(group_expr, projections)
    expr = projection.this if projection and projection.__class__.__name__ == "Alias" else projection
    if expr is None:
        expr = group_expr
    if _is_time_trunc_expr(expr):
        return None
    expression = _sql_name(expr)
    if not expression:
        return None
    hint = {
        "alias": alias,
        "expr": expression,
    }
    column = _column_name(expr)
    if column:
        hint["column"] = column
    return hint


def _is_time_trunc_expr(expr: Any) -> bool:
    try:
        from sqlglot import expressions as exp

        return isinstance(expr, (exp.DateTrunc, exp.DatetimeTrunc, exp.TimeTrunc, exp.TimestampTrunc))
    except Exception:
        return False


def _normalize_time_grain(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    text = str(name if name else value).strip().strip("'\"").lower()
    allowed = {"day", "week", "month", "quarter", "year"}
    return text if text in allowed else ""


def _sql_name(expr: Any) -> str:
    if expr is None:
        return ""
    try:
        return expr.sql(dialect="snowflake")
    except Exception:
        try:
            return expr.sql()
        except Exception:
            return str(expr)


def _column_name(expr: Any) -> str:
    try:
        from sqlglot import expressions as exp

        if isinstance(expr, exp.Column):
            return str(expr.name or "")
    except Exception:
        return ""
    return ""


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


def _dedupe_dimension_expr_hints(values: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen = set()
    for value in values:
        alias = value.get("alias", "")
        expression = value.get("expr", "")
        key = (_normalize_name(alias), _normalize_sql_text(expression))
        if not alias or not expression or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _dedupe_time_group_hints(values: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen = set()
    for value in values:
        alias = value.get("alias", "")
        base_expr = value.get("base_expr", "")
        grain = value.get("grain", "")
        key = (_normalize_name(alias), _normalize_name(base_expr), grain)
        if not grain or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize_sql_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()
