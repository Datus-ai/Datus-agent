# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers for validating generated metrics against source-query shape."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


def extract_metric_queryability_contracts_from_sources(
    source_queries: Iterable[Any],
) -> List[Dict[str, Any]]:
    """Analyze structured SQL through the shared SQL-modeling planner."""
    sources = []
    for index, source_query in enumerate(source_queries, 1):
        if isinstance(source_query, dict):
            source_name = source_query.get("source_sql_name")
            sql = source_query.get("sql")
        else:
            source_name = getattr(source_query, "source_sql_name", None)
            sql = getattr(source_query, "sql", None)
        sources.append(
            {
                "name": str(source_name or f"sql_{index}"),
                "sql": str(sql or ""),
            }
        )
    return _planned_queryability_contracts(sources)


def _planned_queryability_contracts(sources: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not sources:
        return []
    from datus.tools.func_tool.semantic_discovery_tools import analyze_metric_candidate_entries

    result = analyze_metric_candidate_entries(sources)
    if not result.success or not isinstance(result.result, dict):
        return []
    return [
        dict(contract) for contract in result.result.get("queryability_contracts") or [] if isinstance(contract, dict)
    ]


def link_queryability_contracts_to_metric_outputs(
    contracts: Iterable[Dict[str, Any]],
    metric_requirements: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Link source-SQL aliases to stable output IDs before metrics are renamed."""
    requirements_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for requirement in metric_requirements:
        if not isinstance(requirement, dict):
            continue
        output_id = str(requirement.get("output_id") or "").strip()
        preferred_name = str(requirement.get("preferred_name") or "").strip()
        if not output_id or not preferred_name:
            continue
        for source in _source_names(requirement.get("source_sql_name")):
            requirements_by_source.setdefault(source, []).append(requirement)

    linked_contracts: List[Dict[str, Any]] = []
    for contract in contracts:
        normalized_contract = dict(contract)
        hint_names = {
            _normalize_name(hint)
            for hint in contract.get("metric_hints") or []
            if isinstance(hint, str) and hint.strip()
        }
        output_ids: List[str] = []
        for source in _source_names(contract.get("source")):
            for requirement in requirements_by_source.get(source, []):
                if _normalize_name(requirement.get("preferred_name", "")) not in hint_names:
                    continue
                output_id = str(requirement["output_id"])
                if output_id not in output_ids:
                    output_ids.append(output_id)
        if output_ids:
            normalized_contract["metric_output_ids"] = output_ids
        linked_contracts.append(normalized_contract)
    return linked_contracts


def query_backed_queryability_contracts(
    candidate_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build final-grain contracts for SQL-backed datasets.

    Query-backed SQL commonly ends with a passthrough SELECT over a CTE. The
    final SELECT has no GROUP BY of its own, so AST-only extraction cannot see
    the dataset's output grain. The planner already records that grain and the
    stable metric output IDs; use those authoritative contracts here.
    """
    requirements_by_output_id = {
        str(requirement.get("output_id") or "").strip(): requirement
        for requirement in candidate_plan.get("metric_requirements") or []
        if isinstance(requirement, dict) and str(requirement.get("output_id") or "").strip()
    }
    contracts: List[Dict[str, Any]] = []
    for dataset_requirement in candidate_plan.get("dataset_requirements") or []:
        if not isinstance(dataset_requirement, dict):
            continue
        dimension_hints = _dedupe(
            [str(name).strip() for name in dataset_requirement.get("output_grain") or [] if str(name).strip()]
        )
        output_ids = _dedupe(
            [
                str(output_id).strip()
                for output_id in dataset_requirement.get("metric_output_ids") or []
                if str(output_id).strip()
            ]
        )
        if not dimension_hints or not output_ids:
            continue
        metric_hints = _dedupe(
            [
                str(requirements_by_output_id[output_id].get("preferred_name") or "").strip()
                for output_id in output_ids
                if output_id in requirements_by_output_id
                and str(requirements_by_output_id[output_id].get("preferred_name") or "").strip()
            ]
        )
        contracts.append(
            {
                "source": str(dataset_requirement.get("source_sql_name") or "").strip(),
                "dimension_hints": dimension_hints,
                "metric_hints": metric_hints,
                "metric_output_ids": output_ids,
                "contract_source": "query_backed_output_grain",
            }
        )
    return contracts


def summarize_queryability_contracts(contracts: Iterable[Dict[str, Any]]) -> str:
    parts = []
    for contract in contracts:
        dimensions = ", ".join(contract.get("dimension_hints") or [])
        metrics = ", ".join(contract.get("metric_hints") or [])
        if dimensions:
            part = f"{contract.get('source') or 'source SQL'} group-by [{dimensions}] metrics [{metrics}]"
            grains = [
                h.get("grain")
                for h in (contract.get("time_group_hints") or [])
                if isinstance(h, dict) and h.get("grain")
            ]
            if grains:
                part += f" (dry-run with time_granularity='{', '.join(dict.fromkeys(grains))}')"
            parts.append(part)
    return "; ".join(parts)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _source_names(value: Any) -> List[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


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
