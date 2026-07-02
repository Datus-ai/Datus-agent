# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Deterministic description refresh helpers for semantic-model profiles."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

OBSERVED_PROFILE_PREFIX = "Observed profile:"
_OBSERVED_PROFILE_RE = re.compile(r"\s*Observed profile:.*$", re.IGNORECASE | re.DOTALL)


def strip_observed_profile(description: Any) -> str:
    """Return the stable business description without the generated profile suffix."""
    text = " ".join(str(description or "").split())
    return _OBSERVED_PROFILE_RE.sub("", text).strip()


def merge_observed_profile(description: Any, observed_profile: str, *, max_chars: int = 420) -> str:
    """Replace the generated profile suffix while preserving the stable description prefix."""
    base = strip_observed_profile(description).rstrip(" .;")
    observed = " ".join(str(observed_profile or "").split()).strip(" .;")
    if not observed:
        return base

    suffix = f"{OBSERVED_PROFILE_PREFIX} {observed}."
    merged = f"{base}. {suffix}" if base else suffix
    if len(merged) <= max_chars:
        return merged
    if base:
        budget = max_chars - len(base) - len(f". {OBSERVED_PROFILE_PREFIX} ...")
        clipped = _clip_text(observed, max(40, budget)).strip(" .;")
        result = f"{base}. {OBSERVED_PROFILE_PREFIX} {clipped}."
        return _clip_text(result, max_chars) if len(result) > max_chars else result
    return _clip_text(merged, max_chars)


def build_table_observed_profile(table_evidence: Dict[str, Any]) -> str:
    """Summarize table-level SQL/profile evidence for a description."""
    parts: List[str] = []
    distribution = table_evidence.get("data_distribution_profile") or {}
    row_count = _profile_scalar(distribution.get("row_count"))
    if row_count:
        parts.append(f"observed row count {row_count}")

    query_count = _as_int(table_evidence.get("query_count"))
    if query_count:
        parts.append(f"referenced by {query_count} historical quer{'y' if query_count == 1 else 'ies'}")

    duration_phrases = _duration_profile_phrases(distribution.get("date_duration_profiles") or [])
    if duration_phrases:
        parts.append(f"typical durations include {duration_phrases[0]}")

    filter_fields = _business_filter_fields(table_evidence.get("common_business_filter_templates") or [])
    if filter_fields:
        parts.append(f"common filters use {', '.join(filter_fields[:3])}")

    return "; ".join(parts[:4])


def build_column_observed_profile(
    column_profile: Dict[str, Any],
    *,
    field_usage: Optional[Dict[str, Any]] = None,
    join_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Summarize one column's sampled distribution and SQL-history usage."""
    field_usage = field_usage or {}
    parts: List[str] = []
    kind = str(column_profile.get("kind") or "").lower()
    stats = column_profile.get("stats") if isinstance(column_profile.get("stats"), dict) else {}

    if kind in {"categorical", "boolean"}:
        distinct_count = _as_int(stats.get("distinct_count"))
        if distinct_count is not None:
            parts.append(f"{distinct_count} distinct non-null value{'s' if distinct_count != 1 else ''}")
        top_values = [
            _profile_scalar(item.get("value"))
            for item in column_profile.get("top_values") or []
            if isinstance(item, dict) and item.get("value") not in (None, "")
        ]
        if top_values:
            parts.append(f"common values include {', '.join(top_values[:3])}")

    elif kind == "numeric":
        min_value = _profile_scalar(stats.get("min_value"))
        max_value = _profile_scalar(stats.get("max_value"))
        if min_value and max_value:
            parts.append(f"observed range {min_value}-{max_value}")
        percentiles = column_profile.get("percentiles") if isinstance(column_profile.get("percentiles"), dict) else {}
        p50 = _profile_scalar(percentiles.get("p50"))
        p90 = _profile_scalar(percentiles.get("p90"))
        if p50 and p90:
            parts.append(f"p50 {p50}, p90 {p90}")

    elif kind == "temporal":
        min_value = _profile_scalar(stats.get("min_value"))
        max_value = _profile_scalar(stats.get("max_value"))
        if min_value and max_value:
            parts.append(f"observed span {min_value} to {max_value}")
        temporal = (
            column_profile.get("temporal_summary") if isinstance(column_profile.get("temporal_summary"), dict) else {}
        )
        freshness = _as_int(temporal.get("freshness_days_from_profile_date"))
        if freshness is not None:
            parts.append(_freshness_phrase(freshness))

    null_rate = _as_float(stats.get("null_rate"))
    if null_rate is not None and null_rate >= 0.01:
        parts.append(f"null rate {_format_percent(null_rate)}")

    usage_part = _field_usage_phrase(field_usage)
    if usage_part:
        parts.append(usage_part)

    join_part = _join_profile_phrase(join_profile)
    if join_part:
        parts.append(join_part)

    return "; ".join(_dedupe(parts)[:4])


def refresh_metricflow_yaml_descriptions(docs: List[dict], profile_evidence: Dict[str, Any]) -> int:
    """Patch MetricFlow data_source descriptions from profiler evidence."""
    tables = _tables_from_evidence(profile_evidence)
    changed = 0
    for doc in docs:
        data_source = doc.get("data_source") if isinstance(doc, dict) else None
        if not isinstance(data_source, dict):
            continue
        table_key = _match_table_key(data_source.get("sql_table") or data_source.get("name"), tables)
        table_evidence = tables.get(table_key or "")
        if not table_evidence:
            continue
        table_observed = build_table_observed_profile(table_evidence)
        if table_observed:
            changed += _merge_description(data_source, table_observed)
        changed += _refresh_named_items(
            items=list(data_source.get("identifiers") or [])
            + list(data_source.get("dimensions") or [])
            + list(data_source.get("measures") or []),
            table_evidence=table_evidence,
        )
    return changed


def refresh_osi_yaml_descriptions(docs: List[dict], profile_evidence: Dict[str, Any]) -> int:
    """Patch OSI dataset/field descriptions from profiler evidence."""
    tables = _tables_from_evidence(profile_evidence)
    changed = 0
    for dataset in _iter_osi_datasets(docs):
        table_key = _match_table_key(_osi_dataset_table_name(dataset), tables)
        table_evidence = tables.get(table_key or "")
        if not table_evidence:
            continue
        table_observed = build_table_observed_profile(table_evidence)
        if table_observed:
            changed += _merge_description(dataset, table_observed)
        field_items = []
        for key in ("fields", "dimensions", "measures", "identifiers", "columns"):
            value = dataset.get(key)
            if isinstance(value, list):
                field_items.extend(item for item in value if isinstance(item, dict))
        changed += _refresh_named_items(items=field_items, table_evidence=table_evidence)
    return changed


def _refresh_named_items(items: Iterable[dict], table_evidence: Dict[str, Any]) -> int:
    changed = 0
    distribution = table_evidence.get("data_distribution_profile") or {}
    column_profiles = distribution.get("columns") or {}
    field_usage = table_evidence.get("field_usage_statistics") or {}
    join_profiles = distribution.get("join_relationship_profiles") or []
    for item in items:
        name = str(item.get("name") or "")
        expr = str(item.get("expr") or name)
        column_key = _match_column_key(expr, column_profiles) or _match_column_key(name, column_profiles)
        usage_key = _match_column_key(expr, field_usage) or _match_column_key(name, field_usage)
        column_profile = column_profiles.get(column_key or "")
        usage = field_usage.get(usage_key or "") if usage_key else None
        if not isinstance(column_profile, dict):
            continue
        observed = build_column_observed_profile(
            column_profile,
            field_usage=usage if isinstance(usage, dict) else None,
            join_profile=_join_profile_for_column(expr or name, join_profiles),
        )
        if observed:
            changed += _merge_description(item, observed)
    return changed


def _merge_description(item: dict, observed: str) -> int:
    current = item.get("description", "")
    updated = merge_observed_profile(current, observed)
    if updated != current:
        item["description"] = updated
        return 1
    return 0


def _tables_from_evidence(profile_evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tables = profile_evidence.get("tables") if isinstance(profile_evidence, dict) else None
    return {str(key): value for key, value in (tables or {}).items() if isinstance(value, dict)}


def _match_table_key(value: Any, tables: Dict[str, Any]) -> str:
    candidates = _identifier_candidates(value)
    for candidate in candidates:
        for table_name in tables:
            if candidate == table_name.lower() or candidate == table_name.split(".")[-1].lower():
                return table_name
    return ""


def _match_column_key(value: Any, columns: Dict[str, Any]) -> str:
    candidates = _identifier_candidates(value)
    for candidate in candidates:
        for column_name in columns:
            if candidate == column_name.lower():
                return column_name
    return ""


def _identifier_candidates(value: Any) -> List[str]:
    text = str(value or "").strip().strip('`"[]')
    if not text:
        return []
    parts = [part.strip().strip('`"[]') for part in text.split(".") if part.strip()]
    candidates = [text.lower()]
    if parts:
        candidates.append(parts[-1].lower())
    return _dedupe(candidates)


def _iter_osi_datasets(docs: List[dict]) -> Iterable[dict]:
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        semantic_models = doc.get("semantic_model")
        if isinstance(semantic_models, list):
            for semantic_model in semantic_models:
                if isinstance(semantic_model, dict):
                    for dataset in semantic_model.get("datasets") or []:
                        if isinstance(dataset, dict):
                            yield dataset
        for dataset in doc.get("datasets") or []:
            if isinstance(dataset, dict):
                yield dataset


def _osi_dataset_table_name(dataset: dict) -> str:
    source = dataset.get("source")
    if isinstance(source, dict) and source.get("table"):
        return str(source["table"])
    if isinstance(source, str) and source:
        return source
    return str(dataset.get("table") or dataset.get("name") or "")


def _business_filter_fields(templates: List[dict]) -> List[str]:
    fields = []
    for template in templates:
        for field in template.get("fields") or []:
            if field:
                fields.append(str(field))
    return _dedupe(fields)


def _duration_profile_phrases(duration_profiles: List[dict]) -> List[str]:
    phrases = []
    for profile in duration_profiles:
        if not isinstance(profile, dict):
            continue
        left = str(profile.get("left_column") or "")
        right = str(profile.get("right_column") or "")
        deltas = profile.get("delta_days") if isinstance(profile.get("delta_days"), dict) else {}
        p50 = _profile_scalar(deltas.get("p50"))
        p90 = _profile_scalar(deltas.get("p90"))
        if not left or not right or not p50:
            continue
        phrase = f"{left} to {right} p50 {p50} days"
        if p90:
            phrase += f", p90 {p90} days"
        phrases.append(phrase)
    return _dedupe(phrases)


def _freshness_phrase(freshness_days_from_profile_date: int) -> str:
    if freshness_days_from_profile_date == 0:
        return "latest value on profiling date"
    if freshness_days_from_profile_date > 0:
        unit = "day" if freshness_days_from_profile_date == 1 else "days"
        return f"latest value {freshness_days_from_profile_date} {unit} before profiling"
    days_after = abs(freshness_days_from_profile_date)
    unit = "day" if days_after == 1 else "days"
    return f"latest value {days_after} {unit} after profiling"


def _field_usage_phrase(field_usage: Dict[str, Any]) -> str:
    phrases = []
    operators = {str(item).upper() for item in field_usage.get("operators") or []}
    if _as_int(field_usage.get("filter_count")):
        if operators & {"=", "IN", "!="}:
            phrases.append("frequently used as a categorical filter")
        elif operators & {">", ">=", "<", "<=", "BETWEEN"}:
            phrases.append("frequently used as a range filter")
        else:
            phrases.append("frequently filtered")
    if _as_int(field_usage.get("group_by_count")):
        phrases.append("commonly grouped")
    if _as_int(field_usage.get("aggregate_count")):
        phrases.append("used in aggregate expressions")
    return ", ".join(phrases[:2])


def _join_profile_for_column(column_name: str, join_profiles: List[dict]) -> Optional[dict]:
    column_leaf = _identifier_candidates(column_name)
    for profile in join_profiles:
        if not isinstance(profile, dict):
            continue
        source = _identifier_candidates(profile.get("source_column"))
        target = _identifier_candidates(profile.get("target_column"))
        if set(column_leaf) & (set(source) | set(target)):
            return profile
    return None


def _join_profile_phrase(join_profile: Optional[Dict[str, Any]]) -> str:
    if not join_profile:
        return ""
    parts = []
    coverage = _as_float(join_profile.get("referential_coverage"))
    if coverage is not None:
        parts.append(f"referential coverage {_format_percent(coverage)}")
    hint = str(join_profile.get("join_cardinality_hint") or "").replace("_", " ")
    if hint:
        parts.append(hint)
    return ", ".join(parts[:2])


def _as_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_scalar(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _clip_text(value: str, max_chars: int) -> str:
    text = " ".join(str(value).split())
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return text[: max_chars - 3].rstrip(" ,;") + "..."


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
