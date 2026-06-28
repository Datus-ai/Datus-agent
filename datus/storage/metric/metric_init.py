# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
from datus.agent.node.semantic_authoring import (
    AUTHORING_FORMAT_OSI,
    default_osi_semantic_model_file,
    resolve_authoring_format,
)
from datus.configuration.agent_config import AgentConfig
from datus.prompts.prompt_manager import get_prompt_manager
from datus.schemas.action_history import (
    ActionHistory,  # noqa: F401  (forward-ref for action_callback)
    ActionHistoryManager,
    ActionStatus,
)
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.storage.knowledge_provenance import (
    METRIC_ARTIFACT_TYPE,
    KnowledgeProvenanceStore,
    build_metric_provenance_rows,
    is_knowledge_provenance_enabled,
)
from datus.storage.semantic_model.auto_create import (
    ensure_semantic_models_exist,
    extract_table_sql_evidence,
    extract_tables_from_sql_list,
)
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import suppress_keyboard_input

logger = get_logger(__name__)

BIZ_NAME = "metric_init"


def _action_status_value(action: Any) -> Optional[str]:
    status = getattr(action, "status", None)
    if status is None:
        return None
    return status.value if hasattr(status, "value") else str(status)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _parse_context_ids(value: Any) -> list[str]:
    text = _clean_cell(value)
    if not text:
        return []
    parts = [part.strip() for part in text.replace(",", ";").split(";")]
    return [part for part in parts if part]


def _parse_source_metadata(value: Any) -> dict[str, Any]:
    text = _clean_cell(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"raw": text}


def _source_provenance_from_row(row: Any, row_index: int, success_story: str) -> Optional[dict[str, Any]]:
    context_ids: list[str] = []
    for column in ("source_context_ids", "source_context_id", "context_ids", "context_id"):
        context_ids.extend(_parse_context_ids(row.get(column)))
    context_ids = list(dict.fromkeys(context_ids))
    if not context_ids:
        return None

    metadata = _parse_source_metadata(row.get("source_metadata"))
    source_id = _clean_cell(row.get("source_id")) or f"{Path(success_story).name}:{row_index}"
    source_type = _clean_cell(row.get("source_type")) or "success_story"
    metadata.setdefault("source_id", source_id)
    metadata.setdefault("source_type", source_type)
    metadata.setdefault("row_index", row_index)
    question = _clean_cell(row.get("question"))
    if question:
        metadata.setdefault("question", question)
    task_id = _clean_cell(row.get("task_id"))
    if task_id:
        metadata.setdefault("task_id", task_id)

    return {
        "source_id": source_id,
        "source_type": source_type,
        "source_context_ids": context_ids,
        "source_metadata": metadata,
    }


def _extract_metric_artifact_ids(payload: Any) -> list[str]:
    ids: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            text = str(item).strip()
            if text and text not in ids:
                ids.append(text)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            add(value.get("metric_artifact_ids"))
            add(value.get("_synced_metric_artifact_ids"))
            for nested_key in ("result", "sync", "execution_stats", "metric_sync"):
                if nested_key in value:
                    visit(value[nested_key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
    return ids


def _metric_ids_in_storage(agent_config: AgentConfig) -> set[str]:
    try:
        from datus.storage.metric.store import MetricRAG

        return {
            str(row["id"]) for row in MetricRAG(agent_config).search_all_metrics(select_fields=["id"]) if row.get("id")
        }
    except Exception as exc:  # pragma: no cover - defensive fallback for storage readiness issues
        logger.debug("Failed to snapshot metric IDs for provenance fallback: %s", exc)
        return set()


def _clear_metric_provenance(agent_config: AgentConfig) -> int:
    if not is_knowledge_provenance_enabled(agent_config):
        return 0
    try:
        return KnowledgeProvenanceStore(agent_config).delete_for_artifact_type(METRIC_ARTIFACT_TYPE)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to clear metric provenance sidecar: %s", exc)
        return 0


def _sync_metric_provenance(
    agent_config: AgentConfig,
    metric_artifact_ids: list[str],
    source_entries: list[dict[str, Any]],
) -> int:
    if not metric_artifact_ids or not source_entries or not is_knowledge_provenance_enabled(agent_config):
        return 0
    if len(source_entries) != 1:
        logger.warning(
            "Skipping metric provenance sync because source-to-metric attribution is ambiguous for %d source row(s)",
            len(source_entries),
        )
        return 0

    source = source_entries[0]
    items: list[dict[str, Any]] = []
    for artifact_id in metric_artifact_ids:
        items.append({"id": artifact_id, **source})
    rows = build_metric_provenance_rows(items)
    if not rows:
        return 0

    try:
        written = KnowledgeProvenanceStore(agent_config).upsert_many(rows)
        logger.info(
            "Synced %d metric provenance row(s) for %d metric artifact(s)",
            written,
            len(metric_artifact_ids),
        )
        return written
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to sync metric provenance sidecar: %s", exc)
        return 0


DEFAULT_METRICS_BATCH_SIZE = 5
METRICS_NODE_NAME = GenMetricsAgenticNode.NODE_NAME
METRICS_RESPONSE_ACTION_TYPE = f"{METRICS_NODE_NAME}_response"


def _json_dump_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_metric_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _source_names(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    return {str(item).strip() for item in raw_values if str(item).strip()}


def _source_scoped_items(items: Any, batch_sources: set[str]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    scoped: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_sources = _source_names(item.get("source_sql_name") or item.get("source") or item.get("sql_name"))
        if not item_sources or item_sources & batch_sources:
            scoped.append(item)
    return scoped


def _agent_config_dialect(agent_config: AgentConfig) -> str:
    try:
        current_db_config = agent_config.current_db_config()
    except Exception:
        return "snowflake"
    value = getattr(current_db_config, "db_type", "") or getattr(current_db_config, "dialect", "") or ""
    return value if isinstance(value, str) and value.strip() else "snowflake"


def _metrics_node_config(agent_config: AgentConfig) -> dict[str, Any]:
    nodes = getattr(agent_config, "agentic_nodes", None)
    if not isinstance(nodes, dict):
        return {}
    node_config = nodes.get(METRICS_NODE_NAME) or {}
    return node_config if isinstance(node_config, dict) else {}


def _metrics_authoring_format(agent_config: AgentConfig) -> str:
    return resolve_authoring_format(agent_config, _metrics_node_config(agent_config))


def _project_relative_path(agent_config: AgentConfig, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    project_root = Path(str(getattr(agent_config, "project_root", "") or ".")).expanduser()
    return project_root / path


async def _ensure_semantic_models_for_metrics(
    agent_config: AgentConfig,
    success_story: str,
    success_story_records: list[dict[str, Any]],
    sql_list: list[str],
    action_callback: Optional[Callable[["ActionHistory"], None]] = None,
) -> tuple[bool, str, list[str]]:
    if _metrics_authoring_format(agent_config) == AUTHORING_FORMAT_OSI:
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async
        from datus.storage.semantic_model.store import SemanticModelRAG

        target_file = default_osi_semantic_model_file(agent_config)
        target_path = _project_relative_path(agent_config, target_file)
        has_semantic_rows = SemanticModelRAG(agent_config).get_size() > 0
        if has_semantic_rows and target_path.exists():
            logger.info("Reusing existing OSI semantic model file for metric bootstrap: %s", target_file)
            return True, "", []

        logger.info("Generating OSI semantic model for metric bootstrap: %s", target_file)
        success, error = await init_success_story_semantic_model_async(
            agent_config,
            success_story,
            emit=None,
            build_mode="incremental",
            action_callback=action_callback,
        )
        if not success:
            return False, error, []
        if not target_path.exists():
            return False, f"OSI semantic model generation did not create target file: {target_file}", []
        return True, "", [target_file]

    all_tables = extract_tables_from_sql_list(sql_list, agent_config)
    if not all_tables:
        return True, "", []

    logger.info(f"Found {len(all_tables)} tables in success story SQL: {all_tables}")
    sql_evidence_by_table = extract_table_sql_evidence(success_story_records, agent_config)

    success, error, created_tables = await ensure_semantic_models_exist(
        all_tables,
        agent_config,
        emit=None,
        sql_evidence_by_table=sql_evidence_by_table,
    )

    if created_tables:
        logger.info(f"Created semantic models for tables: {created_tables}")
    if error:
        logger.warning(f"Semantic model generation had partial failures: {error}")

    return success, error, created_tables


def _build_existing_metric_catalog(agent_config: AgentConfig) -> list[dict[str, Any]]:
    """Load existing metrics once for the bootstrap run."""
    from datus.storage.metric.store import MetricRAG

    try:
        rows = MetricRAG(agent_config).search_all_metrics()
    except Exception as exc:  # pragma: no cover - defensive; generation can proceed without catalog hints
        logger.warning("Failed to load existing metric catalog; continuing without it: %s", exc)
        return []

    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        normalized = _normalize_metric_name(name)
        if normalized in seen:
            continue
        seen.add(normalized)
        catalog.append(
            {
                "name": name,
                "type": row.get("metric_type") or row.get("type") or "",
                "description": row.get("description") or "",
                "subject_path": row.get("subject_path") or [],
                "semantic_model": row.get("semantic_model_name") or "",
                "semantic_model_name": row.get("semantic_model_name") or "",
                "base_measures": row.get("base_measures") or [],
                "dimensions": row.get("dimensions") or [],
                "entities": row.get("entities") or [],
            }
        )
    return catalog


def _final_metric_count(agent_config: AgentConfig) -> int:
    catalog = _build_existing_metric_catalog(agent_config)
    return len(catalog)


def _build_sql_to_table_lineage(sql_entries: list[dict[str, Any]], agent_config: AgentConfig) -> list[dict[str, Any]]:
    from datus.utils.sql_utils import extract_table_names

    lineage: list[dict[str, Any]] = []
    dialect = _agent_config_dialect(agent_config)
    for entry in sql_entries:
        sql = str(entry.get("sql") or "").strip()
        if not sql:
            continue
        try:
            tables = sorted(extract_table_names(sql, dialect=dialect, ignore_empty=True))
        except Exception as exc:
            lineage.append({"source_sql_name": entry.get("name"), "tables": [], "error": str(exc)})
            continue
        lineage.append({"source_sql_name": entry.get("name"), "tables": tables})
    return lineage


def _build_candidate_plan(
    sql_entries: list[dict[str, Any]],
    existing_metric_catalog_json: str,
    agent_config: AgentConfig,
) -> dict[str, Any]:
    """Run SQL metric candidate extraction once before launching gen_metrics agents."""
    from types import SimpleNamespace

    from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools

    if not sql_entries:
        return {
            "available": True,
            "metric_candidates": [],
            "direct_metric_candidates": [],
            "derived_metric_candidates": [],
            "llm_review_candidates": [],
            "base_measures": [],
            "support_measure_candidates": [],
            "non_metric_evidence": [],
            "identity_metric_references": [],
            "parse_errors": [],
            "query_classification": "manual_review_required",
            "source_classifications": [],
            "derived_datasource_recommendations": [],
            "blocked_direct_metric_candidates": [],
            "literal_mappings": [],
            "time_grain_evidence": [],
            "post_aggregation_constraints": [],
            "modeling_plan": {},
            "summary": "Found 0 metric candidates and 0 base measures from 0 SQL queries",
            "sql_to_table_lineage": [],
        }

    try:
        discovery = SemanticDiscoveryTools(SimpleNamespace(agent_config=agent_config, sub_agent_name=METRICS_NODE_NAME))
        result = discovery.analyze_metric_candidates_from_history(
            sql_entries_json=_json_dump_compact(sql_entries),
            existing_metric_catalog_json=existing_metric_catalog_json,
            sample_sql_queries=len(sql_entries),
        )
        if not result.success:
            logger.warning("Metric candidate extraction failed: %s", result.error)
            return {
                "available": False,
                "error": result.error or "metric candidate extraction failed",
                "sql_to_table_lineage": _build_sql_to_table_lineage(sql_entries, agent_config),
            }
        plan = dict(result.result or {})
        plan["available"] = True
        plan["sql_to_table_lineage"] = _build_sql_to_table_lineage(sql_entries, agent_config)
        plan["derived_metric_candidates"] = _annotate_offset_identity_candidates(plan.get("derived_metric_candidates"))
        return plan
    except Exception as exc:  # pragma: no cover - defensive; agent can still infer from SQL prompt
        logger.warning("Metric candidate extraction raised; continuing without a candidate plan: %s", exc)
        return {
            "available": False,
            "error": str(exc),
            "sql_to_table_lineage": _build_sql_to_table_lineage(sql_entries, agent_config),
        }


def _candidate_plan_for_sources(candidate_plan: dict[str, Any], batch_sources: set[str]) -> dict[str, Any]:
    if not candidate_plan or not candidate_plan.get("available"):
        return candidate_plan or {}

    scoped = dict(candidate_plan)
    for key in (
        "metric_candidates",
        "direct_metric_candidates",
        "derived_metric_candidates",
        "llm_review_candidates",
        "base_measures",
        "support_measure_candidates",
        "non_metric_evidence",
        "identity_metric_references",
        "source_classifications",
        "derived_datasource_recommendations",
        "blocked_direct_metric_candidates",
        "metric_aliases",
        "literal_mappings",
        "time_grain_evidence",
        "post_aggregation_constraints",
        "sql_to_table_lineage",
    ):
        scoped[key] = _source_scoped_items(candidate_plan.get(key), batch_sources)

    scoped["summary"] = (
        f"Batch candidate plan for {len(batch_sources)} SQL source(s): "
        f"{len(scoped.get('direct_metric_candidates') or [])} direct, "
        f"{len(scoped.get('derived_metric_candidates') or [])} derived, "
        f"{len(scoped.get('llm_review_candidates') or [])} LLM review, "
        f"{len(scoped.get('blocked_direct_metric_candidates') or [])} blocked direct candidate(s)."
    )
    return scoped


def _candidate_metric_names(candidate_plan: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("direct_metric_candidates", "derived_metric_candidates", "llm_review_candidates", "metric_candidates"):
        for candidate in candidate_plan.get(key) or []:
            if not isinstance(candidate, dict):
                continue
            name = str(candidate.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def _batch_has_no_metric_candidates(batch_candidate_plan: dict[str, Any]) -> bool:
    """Return True when the batch candidate plan has no actionable metric candidates.

    Skipping the LLM call is safe when the plan is available, there are zero
    direct/derived candidates, and there IS non-metric or identity-reference
    evidence (i.e. the SQL was analysed but produced nothing metric-worthy).
    """
    if not batch_candidate_plan or not batch_candidate_plan.get("available"):
        return False
    if _candidate_metric_names(batch_candidate_plan):
        return False
    has_evidence = bool(
        batch_candidate_plan.get("non_metric_evidence")
        or batch_candidate_plan.get("identity_metric_references")
        or batch_candidate_plan.get("derived_datasource_recommendations")
    )
    return has_evidence


def _candidate_metric_items(candidate_plan: dict[str, Any]) -> list[dict[str, Any]]:
    candidates_by_name: dict[str, dict[str, Any]] = {}
    ordered_names: list[str] = []
    seen: set[str] = set()
    for key in ("direct_metric_candidates", "derived_metric_candidates", "llm_review_candidates", "metric_candidates"):
        for candidate in candidate_plan.get(key) or []:
            if not isinstance(candidate, dict):
                continue
            name = str(candidate.get("name") or "").strip()
            normalized = _normalize_metric_name(name)
            if not normalized:
                continue
            if normalized not in seen:
                seen.add(normalized)
                ordered_names.append(normalized)
                candidates_by_name[normalized] = candidate
                continue
            if not _candidate_has_definition_evidence(
                candidates_by_name[normalized]
            ) and _candidate_has_definition_evidence(candidate):
                candidates_by_name[normalized] = candidate
    return [candidates_by_name[name] for name in ordered_names]


def _normalized_scalar(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalized_metric_type(value: Any) -> str:
    metric_type = _normalized_scalar(value)
    return "measure_proxy" if metric_type == "simple" else metric_type


def _normalized_measure_names(value: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(value, list):
        return names
    for item in value:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name") or item.get("measure") or item.get("expr")
        else:
            continue
        normalized = _normalize_metric_name(name)
        if normalized:
            names.add(normalized)
    return names


def _candidate_has_definition_evidence(candidate: dict[str, Any]) -> bool:
    return any(
        candidate.get(field)
        for field in (
            "metric_type",
            "type",
            "semantic_model",
            "semantic_model_name",
            "base_measures",
            "referenced_metrics",
        )
    )


def _candidate_matches_existing_metric(candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
    if not _candidate_has_definition_evidence(candidate):
        return False

    candidate_type = _normalized_metric_type(candidate.get("metric_type") or candidate.get("type"))
    existing_type = _normalized_metric_type(existing.get("type") or existing.get("metric_type"))
    if candidate_type and existing_type and candidate_type != existing_type:
        return False

    candidate_semantic_model = _normalized_scalar(
        candidate.get("semantic_model_name") or candidate.get("semantic_model")
    )
    existing_semantic_model = _normalized_scalar(existing.get("semantic_model_name") or existing.get("semantic_model"))
    if candidate_semantic_model and existing_semantic_model and candidate_semantic_model != existing_semantic_model:
        return False

    candidate_measures = _normalized_measure_names(candidate.get("base_measures"))
    if not candidate_measures and candidate.get("referenced_metrics"):
        candidate_measures = _normalized_measure_names(candidate.get("referenced_metrics"))
    existing_measures = _normalized_measure_names(existing.get("base_measures"))
    if candidate_measures and existing_measures and candidate_measures != existing_measures:
        return False

    return True


def _all_candidate_metrics_satisfied(
    candidate_plan: dict[str, Any], existing_metric_catalog: list[dict[str, Any]]
) -> bool:
    candidates = _candidate_metric_items(candidate_plan)
    if not candidates:
        return False
    existing_by_name = {
        _normalize_metric_name(item.get("name")): item for item in existing_metric_catalog if item.get("name")
    }
    for candidate in candidates:
        existing = existing_by_name.get(_normalize_metric_name(candidate.get("name")))
        if not existing or not _candidate_matches_existing_metric(candidate, existing):
            return False
    return True


def _offset_grain(offset_window: Any) -> Optional[str]:
    parts = str(offset_window or "").strip().lower().split()
    if len(parts) < 2:
        return None
    unit = parts[1].rstrip("s")
    return unit if unit in {"day", "week", "month", "quarter", "year"} else None


def _is_offset_derived_candidate(candidate: dict[str, Any]) -> bool:
    if _normalized_metric_type(candidate.get("metric_type") or candidate.get("type")) != "derived":
        return False
    inputs = candidate.get("inputs")
    return isinstance(inputs, list) and any(isinstance(item, dict) and item.get("offset_window") for item in inputs)


def _candidate_input_metric_names(candidate: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in candidate.get("inputs") or []:
        if isinstance(item, dict):
            normalized = _normalize_metric_name(item.get("name"))
            if normalized:
                names.add(normalized)
    return names


def _annotate_offset_identity_candidates(derived_candidates: Any) -> list[Any]:
    """Annotate offset identity candidates with their canonical equivalent names.

    Identity candidates mined from SQL keep their historical alias (e.g.
    ``previous_month_activity_count``) as the metric name the agent should
    publish; the canonical ``{base}_previous_{grain}`` variant is recorded in
    ``equivalent_names`` so the completeness check accepts either name without
    tempting the agent to rename the metric. Operates purely on structured
    candidates, independent of any authoring format.
    """
    annotated: list[Any] = []
    for candidate in derived_candidates or []:
        if not isinstance(candidate, dict):
            annotated.append(candidate)
            continue
        if candidate.get("name") and _is_offset_derived_candidate(candidate):
            alias_variants = _offset_identity_alias_candidates(candidate)
            equivalent_names = [str(variant.get("name")).strip() for variant in alias_variants if variant.get("name")]
            if equivalent_names:
                candidate = {**candidate, "equivalent_names": equivalent_names}
        annotated.append(candidate)
    return annotated


def _offset_identity_alias_candidates(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = [item for item in candidate.get("inputs") or [] if isinstance(item, dict) and item.get("offset_window")]
    if len(inputs) != 1:
        return []

    offset_input = inputs[0]
    base_name = _normalize_metric_name(offset_input.get("name"))
    alias = _normalize_metric_name(offset_input.get("alias"))
    offset_window = str(offset_input.get("offset_window") or "").strip()
    grain = _offset_grain(offset_window)
    if not base_name or not alias or not grain:
        return []

    expression = _normalize_metric_name(candidate.get("expression"))
    candidate_name = _normalize_metric_name(candidate.get("name"))
    if expression != alias or candidate_name != alias:
        return []

    equivalent_name = f"{base_name}_previous_{grain}"
    if equivalent_name == candidate_name:
        return []

    equivalent = dict(candidate)
    equivalent["name"] = equivalent_name
    equivalent["expression"] = equivalent_name
    equivalent["source_alias"] = equivalent_name
    equivalent["description"] = str(candidate.get("description") or equivalent_name).replace(
        candidate_name, equivalent_name
    )
    equivalent_inputs: list[dict[str, Any]] = []
    for item in candidate.get("inputs") or []:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        if copied.get("offset_window"):
            copied["alias"] = equivalent_name
        equivalent_inputs.append(copied)
    equivalent["inputs"] = equivalent_inputs
    return [equivalent]


def _unique_metric_catalog_by_name(
    existing_metric_catalog: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in existing_metric_catalog:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_metric_name(item.get("name"))
        if normalized:
            grouped.setdefault(normalized, []).append(item)

    unique = {name: items[0] for name, items in grouped.items() if len(items) == 1}
    ambiguous = {name for name, items in grouped.items() if len(items) > 1}
    return unique, ambiguous


def _missing_offset_derived_candidates(
    candidate_plan: dict[str, Any],
    existing_metric_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return offset derived candidates whose inputs exist but which are absent from the catalog.

    Pure completeness check over structured candidates and the metric catalog;
    it never inspects or renders authoring YAML, so it is independent of the
    authoring format (MetricFlow, OSI, ...).
    """
    if not candidate_plan or not candidate_plan.get("available"):
        return []
    derived_candidates = candidate_plan.get("derived_metric_candidates")
    if not isinstance(derived_candidates, list):
        return []

    existing_by_name, ambiguous_existing_names = _unique_metric_catalog_by_name(existing_metric_catalog)
    missing: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in derived_candidates:
        if not (isinstance(candidate, dict) and candidate.get("name") and _is_offset_derived_candidate(candidate)):
            continue
        normalized_name = _normalize_metric_name(candidate.get("name"))
        if not normalized_name or normalized_name in seen:
            continue
        seen.add(normalized_name)
        # Equivalent names (mined alias vs canonical {base}_previous_{grain})
        # describe one metric; any of them existing satisfies the candidate.
        candidate_names = {normalized_name} | {
            _normalize_metric_name(name) for name in candidate.get("equivalent_names") or [] if name
        }
        if candidate_names & ambiguous_existing_names or candidate_names & set(existing_by_name):
            continue
        input_names = _candidate_input_metric_names(candidate)
        if not input_names or input_names & ambiguous_existing_names:
            continue
        if not input_names <= set(existing_by_name):
            continue
        missing.append(candidate)
    return missing


async def _ensure_offset_derived_metrics(
    candidate_plan: dict[str, Any],
    agent_config: AgentConfig,
    subject_tree: Optional[list],
    extra_instructions: Optional[str],
    event_helper: Optional[BatchEventHelper],
    action_callback: Optional[Callable[["ActionHistory"], None]],
    all_query_records: list[dict[str, Any]],
    total_batches: int,
) -> dict[str, Any]:
    """Verify offset derived candidates landed in the metric store; retry the missing ones once.

    The retry re-runs the regular gen_metrics agent with a focused instruction,
    so the metric YAML is always authored by the LLM in whatever format the
    node is configured for. Candidates still missing afterwards are reported,
    never silently dropped.
    """
    catalog = _build_existing_metric_catalog(agent_config)
    missing = _missing_offset_derived_candidates(candidate_plan, catalog)
    if not missing:
        return {"generated": [], "missing": [], "retried": False}

    missing_names = [str(candidate.get("name")).strip() for candidate in missing]
    missing_sources: set[str] = set()
    for candidate in missing:
        missing_sources |= _source_names(
            candidate.get("source_sql_name") or candidate.get("source") or candidate.get("sql_name")
        )
    retry_records = [record for record in all_query_records if record.get("source_sql_name") in missing_sources]
    retry_queries = [record["query"] for record in retry_records] or [
        "Materialize the missing offset derived metric candidates listed in the candidate plan."
    ]

    retry_plan = dict(candidate_plan)
    retry_plan["derived_metric_candidates"] = missing
    retry_plan["summary"] = f"Offset completeness retry for {len(missing)} derived metric candidate(s)."
    retry_instructions = (
        "Offset completeness retry: the following offset derived metric candidates were mined from the "
        f"success-story SQL but are still missing from the metric store: {', '.join(missing_names)}. "
        "Materialize EVERY one of them as a derived metric over its existing input metrics, including "
        "previous-period identity metrics whose expression is just the offset input alias."
    )
    if extra_instructions:
        retry_instructions = f"{extra_instructions}\n\n{retry_instructions}"

    logger.info("Retrying %d missing offset derived metric(s): %s", len(missing_names), missing_names)
    source_entries = [record["source"] for record in retry_records if record.get("source")]
    metric_ids_before = _metric_ids_in_storage(agent_config) if source_entries else set()
    success, error, batch_result = await _generate_metrics_batch(
        retry_queries,
        total_batches,
        agent_config,
        subject_tree,
        retry_instructions,
        event_helper,
        action_callback,
        candidate_plan_json=_json_dump_compact(retry_plan),
        existing_metric_catalog_json=_json_dump_compact(catalog),
    )
    if not success:
        logger.warning("Offset completeness retry batch failed: %s", error)

    provenance_entries = 0
    if source_entries:
        new_artifact_ids = sorted(_metric_ids_in_storage(agent_config) - metric_ids_before)
        if new_artifact_ids:
            provenance_entries = _sync_metric_provenance(agent_config, new_artifact_ids, source_entries)

    refreshed_catalog = _build_existing_metric_catalog(agent_config)
    still_missing_names = {
        _normalize_metric_name(candidate.get("name"))
        for candidate in _missing_offset_derived_candidates(candidate_plan, refreshed_catalog)
    }
    generated = [name for name in missing_names if _normalize_metric_name(name) not in still_missing_names]
    unresolved = [name for name in missing_names if _normalize_metric_name(name) in still_missing_names]
    if unresolved:
        logger.warning("Offset derived metric(s) still missing after completeness retry: %s", unresolved)
    return {
        "generated": generated,
        "missing": unresolved,
        "retried": True,
        "provenance_entries": provenance_entries,
        "batch_result": batch_result if isinstance(batch_result, dict) else None,
    }


async def _generate_metrics_batch(
    batch_queries: list[str],
    batch_idx: int,
    agent_config: AgentConfig,
    subject_tree: Optional[list],
    extra_instructions: Optional[str],
    event_helper: BatchEventHelper,
    action_callback: Optional[Callable[["ActionHistory"], None]],
    candidate_plan_json: Optional[str] = None,
    existing_metric_catalog_json: Optional[str] = None,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Process a single batch of SQL queries for metrics extraction."""
    batch_message = "Analyze the following SQL queries and extract core metrics:\n\n" + "\n\n---\n\n".join(
        batch_queries
    )

    if existing_metric_catalog_json:
        batch_message += (
            "\n\n## Existing Metric Catalog JSON\n"
            "This catalog was loaded once by the bootstrap host before this batch. "
            "Use it as the initial metric catalog; do not call list_metrics only to rediscover existing metrics.\n"
            f"{existing_metric_catalog_json}"
        )

    if candidate_plan_json:
        batch_message += (
            "\n\n## Precomputed Metric Candidate Plan JSON\n"
            "This plan was mined once from the full success-story SQL set before this batch. "
            "Use it as Phase 1 metric-candidate evidence; do not call analyze_metric_candidates_from_history again "
            "unless this JSON is malformed or insufficient for the requested generation.\n"
            f"{candidate_plan_json}"
        )

    if extra_instructions:
        batch_message = f"{batch_message}\n\n## Additional Instructions\n{extra_instructions}"

    current_db_config = agent_config.current_db_config()
    runtime_db_context_getter = getattr(agent_config, "runtime_db_context", None)
    runtime_db_context = runtime_db_context_getter() if callable(runtime_db_context_getter) else {}
    runtime_db_context = runtime_db_context if isinstance(runtime_db_context, dict) else {}
    latest_prompt_version = get_prompt_manager(agent_config=agent_config).get_latest_version("gen_metrics_system")

    metrics_input = SemanticNodeInput(
        user_message=batch_message,
        catalog=runtime_db_context.get("catalog")
        or runtime_db_context.get("catalog_name")
        or current_db_config.catalog,
        database=runtime_db_context.get("database")
        or runtime_db_context.get("database_name")
        or current_db_config.database,
        db_schema=runtime_db_context.get("schema")
        or runtime_db_context.get("db_schema")
        or runtime_db_context.get("schema_name")
        or current_db_config.schema,
        prompt_version=latest_prompt_version,
    )

    metrics_node = GenMetricsAgenticNode(
        agent_config=agent_config,
        execution_mode="workflow",
        subject_tree=subject_tree,
    )

    action_history_manager = ActionHistoryManager()
    metrics_node.input = metrics_input

    batch_id = f"batch-{batch_idx}"

    try:
        final_result = None
        terminal_error = None
        synced_metric_artifact_ids: list[str] = []
        async for action in metrics_node.execute_stream(action_history_manager):
            if action_callback is not None:
                try:
                    action_callback(action)
                except Exception as cb_exc:  # pragma: no cover - defensive
                    logger.debug("metric action_callback raised: %s", cb_exc)
            if event_helper:
                event_helper.item_processing(
                    item_id=batch_id,
                    action_name="gen_metrics",
                    status=_action_status_value(action),
                    messages=action.messages,
                    output=action.output,
                )
            action_type = getattr(action, "action_type", "")
            for artifact_id in _extract_metric_artifact_ids(getattr(action, "output", None)):
                if artifact_id not in synced_metric_artifact_ids:
                    synced_metric_artifact_ids.append(artifact_id)
            if action.status == ActionStatus.FAILED and action_type == "error":
                terminal_error = action.messages or "Metrics extraction failed"
                logger.error(terminal_error)
                continue
            if action.status == ActionStatus.FAILED and action_type == METRICS_RESPONSE_ACTION_TYPE:
                terminal_error = action.messages or "Metrics extraction failed"
                logger.error(terminal_error)
                continue
            if action.status == ActionStatus.SUCCESS and action_type == METRICS_RESPONSE_ACTION_TYPE and action.output:
                final_result = action.output
                logger.debug(f"Metrics generation action (batch {batch_idx}): {action.messages}")
        if terminal_error:
            return False, terminal_error, None
        if final_result is None:
            return False, "Metrics extraction completed but produced no output", None
        if isinstance(final_result, dict) and synced_metric_artifact_ids:
            final_result["_synced_metric_artifact_ids"] = synced_metric_artifact_ids
        return True, "", final_result
    except Exception as e:
        logger.error(f"Error in metrics extraction (batch {batch_idx}): {e}")
        return False, str(e), None


async def init_success_story_metrics_async(
    agent_config: AgentConfig,
    success_story: str,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
    *,
    build_mode: str = "overwrite",
    action_callback: Optional[Callable[["ActionHistory"], None]] = None,
    batch_size: int = DEFAULT_METRICS_BATCH_SIZE,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """
    Async version: Initialize metrics from success story CSV by batch processing.

    This reads all SQL queries from the CSV and processes them in batches
    to extract core unique metrics (deduplicating aggregation patterns).
    Each batch is processed independently so that one failure does not
    block the rest.

    Args:
        agent_config: Agent configuration
        success_story: Path to success story CSV file
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        extra_instructions: Optional extra instructions for the LLM
        build_mode: ``"overwrite"`` (default) regenerates unconditionally;
            ``"incremental"`` skips the LLM call when the metric store
            already contains entries.
        batch_size: Number of SQL queries per batch (default 5).
    """
    if batch_size <= 0:
        from datus.utils.exceptions import DatusException, ErrorCode

        raise DatusException(
            ErrorCode.STORAGE_INVALID_ARGUMENT, error_message=f"batch_size must be > 0, got {batch_size}"
        )

    event_helper = BatchEventHelper(BIZ_NAME, emit)

    if build_mode == "overwrite":
        from datus.storage.metric.store import MetricRAG

        metric_rag = MetricRAG(agent_config)
        logger.info(
            "[overwrite] Wiping metrics rows for datasource '%s' before re-population",
            metric_rag.datasource_id,
        )
        metric_rag.truncate()
        cleared_provenance = _clear_metric_provenance(agent_config)
        if cleared_provenance:
            logger.info("Cleared %d stale metric provenance row(s)", cleared_provenance)

    df = pd.read_csv(success_story)

    # Emit task started
    event_helper.task_started(total_items=len(df), success_story=success_story)

    # Step 0: Check and create missing semantic models
    success_story_records = []
    sql_entries: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        sql = row.get("sql")
        if not sql:
            continue
        question = row.get("question")
        success_story_records.append({"sql": sql, "question": question})
        sql_entries.append({"name": f"sql_{idx + 1}", "sql": sql, "question": question})
    sql_list = [record["sql"] for record in success_story_records]
    success, error, _created_semantic_models = await _ensure_semantic_models_for_metrics(
        agent_config,
        success_story,
        success_story_records,
        sql_list,
        action_callback=action_callback,
    )

    if not success:
        error_msg = f"Failed to create semantic models: {error}"
        logger.error(error_msg)
        event_helper.task_failed(error=error_msg)
        return False, error_msg, None

    existing_metric_catalog = _build_existing_metric_catalog(agent_config)
    existing_metric_catalog_json = _json_dump_compact(existing_metric_catalog)
    candidate_plan = _build_candidate_plan(sql_entries, existing_metric_catalog_json, agent_config)
    logger.info(
        "Prepared metric bootstrap context: existing_metrics=%d, candidate_plan_available=%s",
        len(existing_metric_catalog),
        candidate_plan.get("available"),
    )

    # Build query records for all rows. Optional source-context columns are only
    # used by benchmark provenance mode and do not affect normal bootstrap data.
    all_query_records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        sql = row["sql"]
        question = row["question"]
        all_query_records.append(
            {
                "source_sql_name": f"sql_{idx + 1}",
                "query": f"Query {idx + 1}:\nQuestion: {question}\nSQL:\n{sql}",
                "source": _source_provenance_from_row(row, idx, success_story),
            }
        )

    # Split into batches
    batches = [all_query_records[i : i + batch_size] for i in range(0, len(all_query_records), batch_size)]
    total_batches = len(batches)

    logger.info(
        f"Processing {len(df)} SQL queries in {total_batches} batch(es) (batch_size={batch_size}) for metrics extraction"
    )

    event_helper.task_processing(total_items=total_batches)

    completed_batches = 0
    failed_batches: list[tuple[int, str]] = []
    merged_result: Optional[dict[str, Any]] = None
    provenance_entries = 0
    skipped_batches = 0

    for batch_idx, batch_records in enumerate(batches):
        batch_queries = [record["query"] for record in batch_records]
        batch_sources = {record["source_sql_name"] for record in batch_records}
        batch_candidate_plan = _candidate_plan_for_sources(candidate_plan, batch_sources)
        batch_candidate_plan_json = _json_dump_compact(batch_candidate_plan) if batch_candidate_plan else ""
        source_entries = [record["source"] for record in batch_records if record.get("source")]

        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_queries)} queries)")

        metric_ids_before = _metric_ids_in_storage(agent_config) if source_entries else set()

        if build_mode == "incremental" and _all_candidate_metrics_satisfied(
            batch_candidate_plan,
            existing_metric_catalog,
        ):
            skipped_batches += 1
            completed_batches += 1
            skipped_names = _candidate_metric_names(batch_candidate_plan)
            logger.info(
                "Skipping metrics batch %d/%d: all %d candidate metric(s) already exist",
                batch_idx + 1,
                total_batches,
                len(skipped_names),
            )
            if event_helper:
                event_helper.item_processing(
                    item_id=f"batch-{batch_idx}",
                    action_name="gen_metrics_skip",
                    status=ActionStatus.SUCCESS.value,
                    messages=(
                        f"Skipped metrics batch {batch_idx + 1}/{total_batches}: "
                        f"existing metrics already satisfy candidates {skipped_names}"
                    ),
                    output={
                        "skipped": True,
                        "skipped_metric_candidates": skipped_names,
                        "batch_index": batch_idx,
                    },
                )

            batch_result = {
                "skipped_batches": 1,
                "skipped_queries": len(batch_records),
                "skipped_metric_candidates": skipped_names,
            }
            if merged_result is None:
                merged_result = batch_result
            elif isinstance(merged_result, dict):
                for key, value in batch_result.items():
                    if key in merged_result and isinstance(merged_result[key], list) and isinstance(value, list):
                        merged_result[key].extend(value)
                    elif key in merged_result and isinstance(merged_result[key], int) and isinstance(value, int):
                        merged_result[key] += value
                    elif key not in merged_result:
                        merged_result[key] = value
            continue

        if _batch_has_no_metric_candidates(batch_candidate_plan):
            skipped_batches += 1
            completed_batches += 1
            non_metric_count = len(batch_candidate_plan.get("non_metric_evidence") or [])
            identity_count = len(batch_candidate_plan.get("identity_metric_references") or [])
            logger.info(
                "Skipping metrics batch %d/%d: no metric candidates (%d non-metric evidence, %d identity references)",
                batch_idx + 1,
                total_batches,
                non_metric_count,
                identity_count,
            )
            if event_helper:
                event_helper.item_processing(
                    item_id=f"batch-{batch_idx}",
                    action_name="gen_metrics_skip",
                    status=ActionStatus.SUCCESS.value,
                    messages=(
                        f"Skipped metrics batch {batch_idx + 1}/{total_batches}: "
                        f"candidate plan contains no metric candidates "
                        f"({non_metric_count} non-metric, {identity_count} identity ref)"
                    ),
                    output={
                        "skipped": True,
                        "skip_reason": "no_metric_candidates",
                        "non_metric_evidence_count": non_metric_count,
                        "identity_reference_count": identity_count,
                        "batch_index": batch_idx,
                    },
                )

            batch_result = {
                "skipped_batches": 1,
                "skipped_queries": len(batch_records),
            }
            if merged_result is None:
                merged_result = batch_result
            elif isinstance(merged_result, dict):
                for key, value in batch_result.items():
                    if key in merged_result and isinstance(merged_result[key], int) and isinstance(value, int):
                        merged_result[key] += value
                    elif key not in merged_result:
                        merged_result[key] = value
            continue

        success, error, batch_result = await _generate_metrics_batch(
            batch_queries,
            batch_idx,
            agent_config,
            subject_tree,
            extra_instructions,
            event_helper,
            action_callback,
            candidate_plan_json=batch_candidate_plan_json,
            existing_metric_catalog_json=existing_metric_catalog_json,
        )

        if success and batch_result is not None:
            completed_batches += 1
            metric_artifact_ids = _extract_metric_artifact_ids(batch_result)
            if source_entries and not metric_artifact_ids:
                metric_artifact_ids = sorted(_metric_ids_in_storage(agent_config) - metric_ids_before)
            batch_provenance_entries = _sync_metric_provenance(agent_config, metric_artifact_ids, source_entries)
            provenance_entries += batch_provenance_entries
            if isinstance(batch_result, dict) and batch_provenance_entries:
                batch_result["provenance_entries"] = (
                    batch_result.get("provenance_entries", 0) + batch_provenance_entries
                )

            if merged_result is None:
                merged_result = batch_result
            elif isinstance(merged_result, dict) and isinstance(batch_result, dict):
                for key, value in batch_result.items():
                    if key in merged_result and isinstance(merged_result[key], list) and isinstance(value, list):
                        merged_result[key].extend(value)
                    elif key in merged_result and isinstance(merged_result[key], int) and isinstance(value, int):
                        merged_result[key] += value
                    elif key not in merged_result:
                        merged_result[key] = value
            if batch_idx + 1 < total_batches:
                existing_metric_catalog = _build_existing_metric_catalog(agent_config)
                existing_metric_catalog_json = _json_dump_compact(existing_metric_catalog)
                candidate_plan = _build_candidate_plan(sql_entries, existing_metric_catalog_json, agent_config)
                logger.info(
                    "Refreshed metric bootstrap context after batch %d/%d: "
                    "existing_metrics=%d, candidate_plan_available=%s",
                    batch_idx + 1,
                    total_batches,
                    len(existing_metric_catalog),
                    candidate_plan.get("available"),
                )
            logger.info(f"Batch {batch_idx + 1}/{total_batches} completed successfully")
        else:
            failed_batches.append((batch_idx, error))
            logger.warning(f"Batch {batch_idx + 1}/{total_batches} failed: {error}, continuing with remaining batches")

    if completed_batches == 0:
        error_summary = "; ".join(f"batch {i + 1}: {e}" for i, e in failed_batches)
        error_msg = f"All {total_batches} batch(es) failed: {error_summary}"
        logger.error(error_msg)
        event_helper.task_failed(error=error_msg)
        return False, error_msg, None

    partial_error = ""
    if failed_batches:
        partial_error = "; ".join(f"batch {i + 1}: {e}" for i, e in failed_batches)
        logger.warning(f"Metrics extraction partially succeeded: {partial_error}")

    if isinstance(merged_result, dict):
        offset_completeness = await _ensure_offset_derived_metrics(
            candidate_plan=candidate_plan,
            agent_config=agent_config,
            subject_tree=subject_tree,
            extra_instructions=extra_instructions,
            event_helper=event_helper,
            action_callback=action_callback,
            all_query_records=all_query_records,
            total_batches=total_batches,
        )
        if offset_completeness.get("retried"):
            provenance_entries += offset_completeness.get("provenance_entries", 0)
            merged_result["offset_retry_generated_metrics"] = offset_completeness.get("generated", [])
        if offset_completeness.get("missing"):
            merged_result["missing_offset_derived_metrics"] = offset_completeness["missing"]
        if provenance_entries:
            merged_result["provenance_entries"] = provenance_entries
        final_metrics_count = _final_metric_count(agent_config)
        merged_result["metrics_count"] = final_metrics_count
        merged_result["final_metrics_count"] = final_metrics_count
        if skipped_batches:
            merged_result["skipped_batches"] = skipped_batches

    logger.info(f"Metrics extraction completed: {completed_batches}/{total_batches} batch(es) succeeded")
    event_helper.task_completed(
        total_items=total_batches,
        completed_items=completed_batches,
        failed_items=len(failed_batches),
    )
    return True, partial_error, merged_result


def init_success_story_metrics(
    agent_config: AgentConfig,
    success_story: str,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
    *,
    build_mode: str = "overwrite",
    batch_size: int = DEFAULT_METRICS_BATCH_SIZE,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """
    Sync wrapper: Initialize metrics from success story CSV by batch processing.

    Args:
        agent_config: Agent configuration
        success_story: Path to success story CSV file
        subject_tree: Optional predefined subject tree categories
        emit: Optional callback to stream BatchEvent progress events
        extra_instructions: Optional extra instructions for the LLM
        build_mode: Forwarded to :func:`init_success_story_metrics_async`.
        batch_size: Number of SQL queries per batch (default 5).
    """
    with suppress_keyboard_input():
        return asyncio.run(
            init_success_story_metrics_async(
                agent_config,
                success_story,
                subject_tree,
                emit,
                extra_instructions,
                build_mode=build_mode,
                batch_size=batch_size,
            )
        )


def init_semantic_yaml_metrics(
    yaml_file_path: str,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """
    Initialize ONLY metrics from semantic YAML file, skip semantic model objects.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
    """
    if not os.path.exists(yaml_file_path):
        logger.error(f"Semantic YAML file {yaml_file_path} not found")
        return False, f"Semantic YAML file {yaml_file_path} not found"

    # Import from semantic_model package to avoid circular dependency
    from datus.storage.semantic_model.semantic_model_init import process_semantic_yaml_file

    return process_semantic_yaml_file(yaml_file_path, agent_config, include_semantic_objects=False)
