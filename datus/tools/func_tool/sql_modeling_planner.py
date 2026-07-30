# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared SQL-modeling preflight for semantic-model and metric authoring."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional

from pydantic import BaseModel, Field

from datus.schemas.semantic_agentic_node_models import SourceQueryEvidence
from datus.tools.func_tool.base import FuncToolResult
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig
    from datus.tools.func_tool.generation_evidence import GenerationEvidence

logger = get_logger(__name__)

SQL_MODELING_PLANNER_VERSION = "1"


class SqlModelingPlan(BaseModel):
    """Request-local modeling evidence shared by both authoring nodes."""

    planner_version: str = SQL_MODELING_PLANNER_VERSION
    source_fingerprint: str
    metric_catalog_fingerprint: str
    source_queries: list[SourceQueryEvidence] = Field(default_factory=list)
    existing_metric_catalog: list[dict[str, Any]] = Field(default_factory=list)
    candidate_plan: dict[str, Any] = Field(default_factory=dict)
    semantic_source_evidence: dict[str, Any] = Field(default_factory=dict)

    def prompt_payload(self) -> dict[str, Any]:
        """Return the stable subset that the authoring model must consume."""
        candidate_plan = copy.deepcopy(self.candidate_plan)
        source_indexes = {source.source_sql_name: index for index, source in enumerate(self.source_queries, 1)}
        for requirement in candidate_plan.get("dataset_requirements") or []:
            if not isinstance(requirement, dict):
                continue
            requirement.pop("sql", None)
            source_index = source_indexes.get(str(requirement.get("source_sql_name") or ""))
            if source_index is not None:
                requirement["source_index"] = source_index
        return {
            "planner_version": self.planner_version,
            "source_fingerprint": self.source_fingerprint,
            "metric_catalog_fingerprint": self.metric_catalog_fingerprint,
            "candidate_plan": candidate_plan,
            "existing_metric_catalog": self.existing_metric_catalog,
            "semantic_source_evidence": self.semantic_source_evidence,
        }


class SqlModelingEntry(BaseModel):
    """Business metadata for one SQL statement owned by the request."""

    source_index: int = Field(..., ge=1, description="1-based SQL position in the current request")
    name: str = Field(..., description="Meaningful English snake_case business name")
    question: str = Field(default="", description="Business question answered by this SQL")


def planned_physical_tables(plan: SqlModelingPlan) -> list[str]:
    """Return unique physical tables extracted by the deterministic planner."""
    tables = []
    seen = set()
    for lineage in plan.candidate_plan.get("sql_to_table_lineage") or []:
        if not isinstance(lineage, dict):
            continue
        for raw_table in lineage.get("tables") or []:
            table = str(raw_table or "").strip()
            identity = table.lower()
            if table and identity not in seen:
                seen.add(identity)
                tables.append(table)
    return tables


def inspect_planned_semantic_sources(
    plan: SqlModelingPlan,
    semantic_discovery_tools: Any,
) -> dict[str, Any]:
    """Run the shared combined physical-source inspection for one SQL plan."""
    tables = planned_physical_tables(plan)
    if not tables:
        return {"status": "not_required", "tables": [], "relationships": []}
    if semantic_discovery_tools is None:
        return {
            "status": "partial",
            "error": "Semantic source inspection is unavailable.",
            "tables": tables,
        }
    inspected = semantic_discovery_tools.inspect_semantic_sources(tables)
    if not inspected.success:
        return {
            "status": "partial",
            "error": inspected.error or "Semantic source inspection failed.",
            "tables": tables,
        }
    return {"status": "ready", **(inspected.result or {})}


class SqlModelingPlanTools:
    """Request-local tool that turns LLM-identified SQL into a deterministic plan."""

    permission_category = "semantic_tools"

    def __init__(
        self,
        *,
        agent_config: "AgentConfig",
        sub_agent_name: str,
        user_message_provider: Callable[[], str],
        generation_evidence: "GenerationEvidence",
        plan_consumer: Callable[[Optional[SqlModelingPlan]], None],
        semantic_source_inspector: Optional[Callable[[SqlModelingPlan], dict[str, Any]]] = None,
    ):
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.user_message_provider = user_message_provider
        self.generation_evidence = generation_evidence
        self.plan_consumer = plan_consumer
        self.semantic_source_inspector = semantic_source_inspector
        self._plan: Optional[SqlModelingPlan] = None

    def reset(self) -> None:
        """Clear request-local state when a reusable node starts a new run."""
        self._plan = None

    def request_contains_sql(self) -> bool:
        """Return whether the current request requires SQL modeling preflight."""
        return bool(
            _extract_sql_snippets(
                self.user_message_provider() or "",
                dialect=_agent_config_dialect(self.agent_config),
            )
        )

    def require_plan_for_sql_request(self) -> bool:
        """Require a ready plan at the terminal boundary of SQL-backed requests."""
        if not self.request_contains_sql():
            return False
        self.generation_evidence.require_sql_modeling_plan()
        return True

    def prepare_sql_modeling_plan(
        self,
        sql_entries: List[SqlModelingEntry],
    ) -> FuncToolResult:
        """Analyze every SQL statement extracted from the current request.

        The tool owns the exact SQL text. Identify each statement by its 1-based
        position and provide only a meaningful business name and question.
        Submit all entries in one call. Do not call this tool when the request
        contains no SQL.

        Args:
            sql_entries: Business metadata for every SQL statement in the request.
        """
        try:
            entries = [SqlModelingEntry.model_validate(item) for item in sql_entries or []]
        except Exception as exc:
            return FuncToolResult(success=0, error=f"Invalid sql_entries: {exc}")

        if not entries:
            if self.request_contains_sql():
                self.generation_evidence.set_sql_modeling_plan("unresolved")
                return FuncToolResult(
                    success=0,
                    error="The current request contains SQL. Submit one indexed entry for every statement.",
                    result={"status": "unresolved"},
                )
            return FuncToolResult(
                success=0,
                error="The current request contains no SQL. Skip prepare_sql_modeling_plan.",
            )

        source_sql = _extract_sql_snippets(
            self.user_message_provider() or "",
            dialect=_agent_config_dialect(self.agent_config),
        )
        validation_error = self._validate_entries(entries, source_sql)
        if validation_error:
            self.generation_evidence.set_sql_modeling_plan("unresolved")
            return FuncToolResult(success=0, error=validation_error, result={"status": "unresolved"})

        sources = [
            SourceQueryEvidence(
                source_sql_name=_normalize_business_name(entry.name),
                sql=source_sql[entry.source_index - 1],
                question=entry.question,
                source_type="prompt",
            )
            for entry in sorted(entries, key=lambda item: item.source_index)
        ]
        if self._plan is not None:
            source_fingerprint = _fingerprint_sources(_deduplicate_sources(sources))
            if self._plan.source_fingerprint != source_fingerprint:
                return FuncToolResult(
                    success=0,
                    error=(
                        "The SQL modeling plan is already fixed for this request. "
                        "Reuse the existing plan instead of submitting different SQL."
                    ),
                    result={"status": "unresolved"},
                )
            return FuncToolResult(result={"status": "ready", **self._plan.prompt_payload()})

        plan = SqlModelingPlanner(self.agent_config, self.sub_agent_name).plan(sources)
        if not plan.candidate_plan.get("available", False):
            self.generation_evidence.set_sql_modeling_plan("unresolved", plan.source_fingerprint)
            return FuncToolResult(
                success=0,
                error=str(plan.candidate_plan.get("error") or "SQL modeling analysis failed"),
                result={"status": "unresolved", **plan.prompt_payload()},
            )

        if self.semantic_source_inspector is not None:
            try:
                plan.semantic_source_evidence = self.semantic_source_inspector(plan) or {}
            except Exception as exc:  # schema discovery can be retried explicitly by the authoring model
                logger.warning("Automatic semantic source inspection failed: %s", exc)
                plan.semantic_source_evidence = {
                    "status": "partial",
                    "error": str(exc),
                    "instruction": "Call inspect_semantic_sources once with the required physical tables.",
                }

        self._plan = plan
        self.generation_evidence.set_sql_modeling_plan("ready", plan.source_fingerprint)
        self.generation_evidence.set_metric_queryability_contracts(
            plan.candidate_plan.get("queryability_contracts") or []
        )
        self.generation_evidence.set_required_metric_outputs(plan.candidate_plan.get("metric_requirements") or [])
        self.generation_evidence.set_required_query_backed_datasets(
            plan.candidate_plan.get("dataset_requirements") or []
        )
        self.plan_consumer(plan)
        return FuncToolResult(result={"status": "ready", **plan.prompt_payload()})

    def _validate_entries(self, entries: list[SqlModelingEntry], source_sql: list[str]) -> str:
        names: set[str] = set()
        indexes: set[int] = set()
        expected_indexes = set(range(1, len(source_sql) + 1))

        for index, entry in enumerate(entries, 1):
            name = _normalize_business_name(entry.name)
            if not name or re.fullmatch(r"(?:sql|query|case|statement|item)_?\d*", name):
                return f"sql_entries[{index - 1}].name must be a meaningful English snake_case name."
            if name in names:
                return f"Duplicate SQL business name: {name!r}."
            names.add(name)

            if entry.source_index in indexes:
                return f"Duplicate SQL source_index: {entry.source_index}."
            indexes.add(entry.source_index)

        if indexes != expected_indexes:
            missing = sorted(expected_indexes - indexes)
            unexpected = sorted(indexes - expected_indexes)
            return (
                "sql_entries must identify every SQL statement exactly once. "
                f"missing source_index={missing}, unexpected source_index={unexpected}."
            )
        return ""


def _normalize_business_name(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "").strip())
    return re.sub(r"_+", "_", text).strip("_").lower()


def _normalize_sql_for_source_check(value: str) -> str:
    """Normalize transport line endings while preserving exact SQL content."""
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _extract_sql_snippets(value: str, *, dialect: str = "") -> list[str]:
    """Extract exact request-local SQL without asking the model to reproduce it."""
    from datus.tools.func_tool.metric_queryability import extract_sql_snippets

    return extract_sql_snippets(
        _normalize_sql_for_source_check(value),
        preserve_source=True,
        dialect=dialect,
    )


def source_query_from_success_story_row(
    row: Any,
    row_index: int,
    success_story: str,
) -> Optional[SourceQueryEvidence]:
    """Build identical structured SQL evidence for every bootstrap wrapper."""
    sql = _clean_tabular_cell(row.get("sql"))
    if not sql:
        return None
    provenance = source_provenance_from_success_story_row(row, row_index, success_story) or {}
    return SourceQueryEvidence(
        source_sql_name=f"sql_{row_index + 1}",
        sql=sql,
        question=_clean_tabular_cell(row.get("question")),
        source_id=provenance.get("source_id")
        or _clean_tabular_cell(row.get("source_id"))
        or f"{Path(success_story).name}:{row_index}",
        source_type=(provenance.get("source_type") or _clean_tabular_cell(row.get("source_type")) or "success_story"),
        source_context_ids=provenance.get("source_context_ids", []),
        source_metadata=(provenance.get("source_metadata") or _parse_source_metadata(row.get("source_metadata"))),
    )


def source_provenance_from_success_story_row(
    row: Any,
    row_index: int,
    success_story: str,
) -> Optional[dict[str, Any]]:
    """Normalize optional provenance without requiring external knowledge."""
    context_ids: list[str] = []
    for column in ("source_context_ids", "source_context_id", "context_ids", "context_id"):
        context_ids.extend(_parse_context_ids(row.get(column)))
    context_ids = list(dict.fromkeys(context_ids))
    if not context_ids:
        return None

    metadata = _parse_source_metadata(row.get("source_metadata"))
    source_id = _clean_tabular_cell(row.get("source_id")) or f"{Path(success_story).name}:{row_index}"
    source_type = _clean_tabular_cell(row.get("source_type")) or "success_story"
    metadata.setdefault("source_id", source_id)
    metadata.setdefault("source_type", source_type)
    metadata.setdefault("row_index", row_index)
    question = _clean_tabular_cell(row.get("question"))
    if question:
        metadata.setdefault("question", question)
    task_id = _clean_tabular_cell(row.get("task_id"))
    if task_id:
        metadata.setdefault("task_id", task_id)
    return {
        "source_id": source_id,
        "source_type": source_type,
        "source_context_ids": context_ids,
        "source_metadata": metadata,
    }


def load_existing_metric_catalog(agent_config: "AgentConfig") -> list[dict[str, Any]]:
    """Load the current datasource metric catalog for planning and reuse."""
    from datus.storage.metric.store import MetricRAG

    try:
        rows = MetricRAG(agent_config).search_all_metrics()
    except Exception as exc:  # pragma: no cover - authoring can continue without reuse hints
        logger.warning("Failed to load existing metric catalog; continuing without it: %s", exc)
        return []

    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("name") or "").strip()
        normalized = name.lower()
        if not normalized or normalized in seen:
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


class SqlModelingPlanner:
    """Build one deterministic modeling plan from authoritative SQL evidence."""

    def __init__(self, agent_config: "AgentConfig", sub_agent_name: str):
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name

    def plan(
        self,
        source_queries: Iterable[SourceQueryEvidence],
        existing_metric_catalog: Optional[list[dict[str, Any]]] = None,
    ) -> SqlModelingPlan:
        """Analyze source SQL and return a versioned request-local plan."""
        sources = _deduplicate_sources(source_queries)
        metric_catalog = (
            load_existing_metric_catalog(self.agent_config)
            if existing_metric_catalog is None
            else list(existing_metric_catalog)
        )
        candidate_plan = self._analyze_metric_candidates(sources, metric_catalog)
        return SqlModelingPlan(
            source_fingerprint=_fingerprint_sources(sources),
            metric_catalog_fingerprint=_fingerprint_json(metric_catalog),
            source_queries=sources,
            existing_metric_catalog=metric_catalog,
            candidate_plan=candidate_plan,
        )

    def _analyze_metric_candidates(
        self,
        sources: list[SourceQueryEvidence],
        metric_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from datus.tools.func_tool.semantic_discovery_tools import analyze_metric_candidate_entries

        entries = [_source_entry(source) for source in sources]
        result = analyze_metric_candidate_entries(
            entries,
            metric_catalog,
            agent_config=self.agent_config,
            sub_agent_name=self.sub_agent_name,
        )
        if not result.success:
            return {
                "available": False,
                "error": result.error or "SQL modeling analysis failed",
                "sql_to_table_lineage": self._sql_to_table_lineage(entries),
            }
        plan = dict(result.result or {})
        parse_errors = [item for item in plan.get("parse_errors") or [] if isinstance(item, dict)]
        if parse_errors:
            failed_sources = [
                str(item.get("source") or item.get("source_sql_name") or "<unknown>") for item in parse_errors
            ]
            plan["available"] = False
            plan["error"] = (
                "SQL modeling analysis could not parse every submitted statement. "
                f"Unresolved sources: {', '.join(failed_sources)}."
            )
            plan["sql_to_table_lineage"] = self._sql_to_table_lineage(entries)
            return plan
        plan["available"] = True
        plan["sql_to_table_lineage"] = self._sql_to_table_lineage(entries)
        return plan

    def _sql_to_table_lineage(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from datus.utils.sql_utils import extract_table_names

        dialect = _agent_config_dialect(self.agent_config)
        lineage: list[dict[str, Any]] = []
        for entry in entries:
            sql = str(entry.get("sql") or "").strip()
            try:
                tables = sorted(extract_table_names(sql, dialect=dialect, ignore_empty=True))
                lineage.append({"source_sql_name": entry["name"], "tables": tables})
            except Exception as exc:
                lineage.append({"source_sql_name": entry["name"], "tables": [], "error": str(exc)})
        return lineage


def _source_entry(source: SourceQueryEvidence) -> dict[str, Any]:
    context_id = next((item for item in source.source_context_ids if str(item).strip()), "")
    return {
        "name": source.source_sql_name,
        "sql": source.sql,
        "question": source.question,
        "source_id": source.source_id,
        "source_type": source.source_type,
        "source_context_id": context_id,
        "source_context_ids": source.source_context_ids,
        "source_metadata": source.source_metadata,
    }


def _deduplicate_sources(sources: Iterable[SourceQueryEvidence]) -> list[SourceQueryEvidence]:
    deduplicated: list[SourceQueryEvidence] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        key = (source.source_sql_name.strip(), source.sql.strip())
        if not key[1] or key in seen:
            continue
        seen.add(key)
        deduplicated.append(source)
    return deduplicated


def _fingerprint_sources(sources: list[SourceQueryEvidence]) -> str:
    payload = [
        {
            "name": source.source_sql_name,
            "sql": source.sql,
            "question": source.question,
            "source_id": source.source_id,
            "source_context_ids": source.source_context_ids,
        }
        for source in sources
    ]
    return _fingerprint_json(payload)


def _fingerprint_json(value: Any) -> str:
    return hashlib.sha256(_compact_json(value).encode("utf-8")).hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _clean_tabular_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        missing = value != value
        if bool(missing):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _parse_context_ids(value: Any) -> list[str]:
    text = _clean_tabular_cell(value)
    if not text:
        return []
    parts = [part.strip() for part in text.replace(",", ";").split(";")]
    return [part for part in parts if part]


def _parse_source_metadata(value: Any) -> dict[str, Any]:
    text = _clean_tabular_cell(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"raw": text}


def _agent_config_dialect(agent_config: "AgentConfig") -> str:
    try:
        current_db_config = agent_config.current_db_config()
    except Exception:
        return "snowflake"
    value = getattr(current_db_config, "type", "")
    value = getattr(value, "value", value)
    return value if isinstance(value, str) and value.strip() else "snowflake"
