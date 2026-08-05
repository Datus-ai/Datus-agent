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
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional

from agents import FunctionTool
from pydantic import BaseModel, Field

from datus.schemas.semantic_agentic_node_models import SourceQueryEvidence
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
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
        """Return only source evidence and the editable output plan to the model."""
        return {
            "planner_version": self.planner_version,
            "sources": [
                {
                    "source_id": source.source_sql_name,
                    "source_index": index,
                    "question": source.question,
                    "source_sql": source.sql,
                }
                for index, source in enumerate(self.source_queries, 1)
            ],
            "candidate_plan": copy.deepcopy(self.candidate_plan),
        }


class SqlModelingEntry(BaseModel):
    """Verbatim user-provided SQL plus the business metadata needed to model it."""

    source_index: int = Field(..., ge=1, description="SQL source order across all submitted batches")
    name: str = Field(
        default="",
        description="Optional business label; a stable source name is generated when omitted or repeated",
    )
    question: str = Field(default="", description="Business question answered by this SQL")
    sql: str = Field(..., min_length=1, description="SQL copied verbatim from user input or a read_file result")


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
        generation_evidence: "GenerationEvidence",
        plan_consumer: Callable[[Optional[SqlModelingPlan]], None],
        semantic_source_inspector: Optional[Callable[[SqlModelingPlan], dict[str, Any]]] = None,
    ):
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.generation_evidence = generation_evidence
        self.plan_consumer = plan_consumer
        self.semantic_source_inspector = semantic_source_inspector
        self._pending_entries: dict[int, SqlModelingEntry] = {}
        self._plan: Optional[SqlModelingPlan] = None
        self._baseline_output_ids: list[str] = []
        self._baseline_contract_ids: list[str] = []

    @staticmethod
    def all_tools_name() -> list[str]:
        """Return the complete permission-registry surface for this tool group."""
        return ["prepare_sql_modeling_plan", "update_sql_modeling_plan"]

    def available_tools(self) -> list[FunctionTool]:
        """Expose SQL preflight through the standard native tool-group contract."""
        return [
            trans_to_function_tool(self.prepare_sql_modeling_plan),
            trans_to_function_tool(self.update_sql_modeling_plan, strict_mode=False),
        ]

    def reset(self) -> None:
        """Clear request-local state when a reusable node starts a new run."""
        self._pending_entries.clear()
        self._plan = None
        self._baseline_output_ids = []
        self._baseline_contract_ids = []

    def prepare_sql_modeling_plan(
        self,
        sql_entries: List[SqlModelingEntry],
        finalize: bool = True,
    ) -> FuncToolResult:
        """Analyze every SQL statement identified from user-provided content.

        Copy each complete SQL statement verbatim from the request or from a
        read_file result for a user-specified path. Attach its position and optional
        business name and question. Small inputs should be submitted in one call.
        Large inputs may be split across calls with ``finalize=False``;
        set ``finalize=True`` on the last batch. Do not call this tool when no SQL
        was provided or referenced.

        Args:
            sql_entries: Verbatim SQL and business metadata for every statement.
            finalize: Analyze all collected entries now. Set false only when more
                SQL batches will follow.
        """
        if self._plan is not None:
            return self._handle_finalized_plan_call(sql_entries)

        self.generation_evidence.mark_sql_modeling_preflight_attempted()
        try:
            entries = [SqlModelingEntry.model_validate(item) for item in sql_entries or []]
        except Exception as exc:
            return FuncToolResult(
                success=0,
                error=f"Invalid sql_entries: {exc}",
                result={"status": "unresolved"},
            )

        if not entries and not self._pending_entries:
            return FuncToolResult(
                success=0,
                error="sql_entries must contain every SQL statement identified from the user-provided content.",
                result={"status": "unresolved"},
            )

        validation_error = self._validate_entries(entries)
        if validation_error:
            return FuncToolResult(success=0, error=validation_error, result={"status": "unresolved"})

        conflicting_indexes = sorted(
            entry.source_index
            for entry in entries
            if entry.source_index in self._pending_entries
            and self._pending_entries[entry.source_index].model_dump() != entry.model_dump()
        )
        if conflicting_indexes:
            return FuncToolResult(
                success=0,
                error=(
                    "These SQL source_index values were already collected with different content: "
                    f"{conflicting_indexes}. Continue numbering across batches; previously collected SQL was unchanged."
                ),
                result={
                    "status": "unresolved",
                    "conflicting_source_indexes": conflicting_indexes,
                },
            )

        pending_entries = dict(self._pending_entries)
        pending_entries.update((entry.source_index, entry) for entry in entries)
        validation_error = self._validate_entries(list(pending_entries.values()))
        if validation_error:
            return FuncToolResult(success=0, error=validation_error, result={"status": "unresolved"})

        self._pending_entries = pending_entries
        if not finalize:
            return FuncToolResult(
                result={
                    "status": "collecting",
                    "received_count": len(self._pending_entries),
                    "source_indexes": sorted(self._pending_entries),
                }
            )

        sources = []
        used_source_names: set[str] = set()
        for entry in sorted(self._pending_entries.values(), key=lambda item: item.source_index):
            source_name = _normalize_business_name(entry.name)
            if not source_name or source_name in used_source_names:
                source_name = f"sql_{entry.source_index}"
            suffix = 2
            unique_name = source_name
            while unique_name in used_source_names:
                unique_name = f"{source_name}_{suffix}"
                suffix += 1
            used_source_names.add(unique_name)
            sources.append(
                SourceQueryEvidence(
                    source_sql_name=unique_name,
                    sql=entry.sql,
                    question=entry.question,
                    source_type="prompt",
                )
            )

        plan = SqlModelingPlanner(self.agent_config, self.sub_agent_name).plan(sources)
        if not plan.candidate_plan.get("available", False):
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
        self._capture_plan_baseline(plan)
        self.generation_evidence.mark_sql_modeling_plan_ready(plan.source_fingerprint)
        self.generation_evidence.set_metric_queryability_contracts(
            plan.candidate_plan.get("queryability_contracts") or []
        )
        self.generation_evidence.set_required_metric_outputs(plan.candidate_plan.get("outputs") or [])
        self.plan_consumer(plan)
        status = str(plan.candidate_plan.get("planning_status") or "ready")
        return FuncToolResult(result={"status": status, **plan.prompt_payload()})

    def update_sql_modeling_plan(self, candidate_plan: Dict[str, Any]) -> FuncToolResult:
        """Replace the publish-pending candidate plan while retaining source SQL.

        Use this after authoring, semantic compilation, or warehouse preflight
        reveals a wrong dataset choice, field qualification, time grain, metric
        reuse decision, output binding, or queryability contract. The submitted
        candidate plan is a complete replacement, not a patch. Every original
        ``output_id`` must remain, while its role, expression, status, generated
        SQL, and later metric binding may change. Original SQL remains immutable
        source evidence.

        Args:
            candidate_plan: Complete revised candidate plan returned from the
                initial preflight, with any required corrections applied.
        """
        if self._plan is None:
            return FuncToolResult(
                success=0,
                error="prepare_sql_modeling_plan must complete before the plan can be updated.",
                result={"status": "unresolved"},
            )
        if self.generation_evidence.kb_sync_passed:
            return FuncToolResult(
                success=0,
                error="The SQL modeling plan cannot change after generated artifacts were published.",
                result={"status": "published", **self._plan.prompt_payload()},
            )
        if not isinstance(candidate_plan, dict):
            return FuncToolResult(
                success=0,
                error="candidate_plan must be a complete JSON object.",
                result={"status": "unresolved", **self._plan.prompt_payload()},
            )

        try:
            revised_candidate = self._normalize_revised_candidate_plan(candidate_plan)
        except ValueError as exc:
            return FuncToolResult(
                success=0,
                error=str(exc),
                result={"status": "unresolved", **self._plan.prompt_payload()},
            )

        revised_output_ids = _metric_output_ids(revised_candidate)
        missing_output_ids = [
            output_id for output_id in self._baseline_output_ids if output_id not in set(revised_output_ids)
        ]
        if missing_output_ids:
            return FuncToolResult(
                success=0,
                error=(
                    "A revised SQL modeling plan cannot silently remove original metric outputs. "
                    f"Missing output_ids: {', '.join(missing_output_ids)}."
                ),
                result={"status": "unresolved", **self._plan.prompt_payload()},
            )
        revised_contract_ids = _queryability_contract_ids(revised_candidate)
        missing_contract_ids = [
            contract_id for contract_id in self._baseline_contract_ids if contract_id not in set(revised_contract_ids)
        ]
        if missing_contract_ids:
            return FuncToolResult(
                success=0,
                error=(
                    "A revised SQL modeling plan cannot remove required GROUP BY checks. "
                    f"Missing contract_ids: {', '.join(missing_contract_ids)}."
                ),
                result={"status": "unresolved", **self._plan.prompt_payload()},
            )

        revised_plan = self._plan.model_copy(update={"candidate_plan": revised_candidate}, deep=True)
        self.generation_evidence.invalidate_plan_evidence()
        self.generation_evidence.set_metric_queryability_contracts(
            revised_candidate.get("queryability_contracts") or []
        )
        self.generation_evidence.set_required_metric_outputs(revised_candidate.get("outputs") or [])
        self._plan = revised_plan
        self.plan_consumer(revised_plan)
        status = str(revised_candidate.get("planning_status") or "ready")
        return FuncToolResult(result={"status": status, "updated": True, **revised_plan.prompt_payload()})

    def _capture_plan_baseline(self, plan: SqlModelingPlan) -> None:
        self._baseline_output_ids = _metric_output_ids(plan.candidate_plan)
        self._baseline_contract_ids = _queryability_contract_ids(plan.candidate_plan)

    def _normalize_revised_candidate_plan(self, candidate_plan: Dict[str, Any]) -> Dict[str, Any]:
        if candidate_plan.get("available") is False:
            raise ValueError("candidate_plan.available must remain true for an update.")
        outputs = candidate_plan.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError("candidate_plan.outputs must be a JSON array.")
        normalized_outputs = []
        seen_output_ids: set[str] = set()
        known_sources = {source.source_sql_name for source in self._plan.source_queries}
        for index, raw_output in enumerate(outputs):
            if not isinstance(raw_output, dict):
                raise ValueError(f"candidate_plan.outputs[{index}] must be a JSON object.")
            output = {
                key: copy.deepcopy(raw_output[key])
                for key in ("output_id", "source_id", "name", "expression", "role", "status", "reason")
                if key in raw_output
            }
            output_id = str(output.get("output_id") or "").strip()
            if not output_id:
                raise ValueError(f"candidate_plan.outputs[{index}].output_id is required.")
            if output_id in seen_output_ids:
                raise ValueError(f"candidate_plan.outputs contains duplicate output_id {output_id!r}.")
            seen_output_ids.add(output_id)
            output["output_id"] = output_id
            output["role"] = str(output.get("role") or "metric").strip().lower()
            source_id = str(output.get("source_id") or "").strip()
            if source_id and source_id not in known_sources:
                raise ValueError(f"Output {output_id!r} has unknown source_id {source_id!r}.")
            if source_id:
                output["source_id"] = source_id
            normalized_outputs.append(output)

        revised = {
            "available": True,
            "planning_status": str(candidate_plan.get("planning_status") or "ready"),
            "outputs": normalized_outputs,
            "queryability_contracts": self._normalize_revised_queryability_contracts(
                candidate_plan.get("queryability_contracts"),
                normalized_outputs,
            ),
        }
        generated_sql = candidate_plan.get("generated_sql")
        if generated_sql is not None:
            if not isinstance(generated_sql, dict):
                raise ValueError("candidate_plan.generated_sql must be a JSON object keyed by source_id.")
            known_sources = {source.source_sql_name for source in self._plan.source_queries}
            unknown_sources = sorted(str(key) for key in generated_sql if str(key) not in known_sources)
            if unknown_sources:
                raise ValueError(f"candidate_plan.generated_sql has unknown source_ids: {', '.join(unknown_sources)}.")
            revised["generated_sql"] = {
                str(source_id): str(sql) for source_id, sql in generated_sql.items() if str(sql).strip()
            }
        for key in ("parse_errors", "unresolved_sources", "summary"):
            if key in candidate_plan:
                revised[key] = copy.deepcopy(candidate_plan[key])
        return revised

    def _normalize_revised_queryability_contracts(
        self,
        raw_contracts: Any,
        outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(raw_contracts, list):
            raise ValueError("candidate_plan.queryability_contracts must be a JSON array.")
        known_sources = {source.source_sql_name for source in self._plan.source_queries}
        known_outputs = {
            str(output.get("output_id") or "").strip()
            for output in outputs
            if str(output.get("output_id") or "").strip()
        }
        metric_outputs = {
            str(output.get("output_id") or "").strip()
            for output in outputs
            if str(output.get("role") or "metric").strip().lower() == "metric"
            and str(output.get("status") or "").strip().lower() not in {"ignored", "skipped", "blocked"}
        }
        normalized = []
        seen_contract_ids: set[str] = set()
        for index, raw_contract in enumerate(raw_contracts):
            if not isinstance(raw_contract, dict):
                raise ValueError(f"candidate_plan.queryability_contracts[{index}] must be a JSON object.")
            contract_id = str(raw_contract.get("contract_id") or "").strip()
            source_id = str(raw_contract.get("source_id") or "").strip()
            if not contract_id or not source_id:
                raise ValueError(f"candidate_plan.queryability_contracts[{index}] requires contract_id and source_id.")
            if contract_id in seen_contract_ids:
                raise ValueError(
                    f"candidate_plan.queryability_contracts contains duplicate contract_id {contract_id!r}."
                )
            seen_contract_ids.add(contract_id)
            if source_id not in known_sources:
                raise ValueError(f"Queryability contract {contract_id!r} has unknown source_id {source_id!r}.")
            metric_output_ids = [
                str(output_id).strip()
                for output_id in raw_contract.get("metric_output_ids") or []
                if str(output_id).strip()
            ]
            unknown_outputs = sorted(set(metric_output_ids) - known_outputs)
            if unknown_outputs:
                raise ValueError(
                    f"Queryability contract {contract_id!r} has unknown output_ids: {', '.join(unknown_outputs)}."
                )
            if not metric_output_ids:
                raise ValueError(f"Queryability contract {contract_id!r} requires metric_output_ids.")
            non_metric_outputs = sorted(set(metric_output_ids) - metric_outputs)
            if non_metric_outputs:
                raise ValueError(
                    f"Queryability contract {contract_id!r} references non-metric output_ids: "
                    f"{', '.join(non_metric_outputs)}."
                )
            dimensions = [
                str(dimension).strip() for dimension in raw_contract.get("dimensions") or [] if str(dimension).strip()
            ]
            if not dimensions:
                raise ValueError(f"Queryability contract {contract_id!r} requires a complete dimensions list.")
            contract: Dict[str, Any] = {
                "contract_id": contract_id,
                "source_id": source_id,
                "metric_output_ids": list(dict.fromkeys(metric_output_ids)),
                "dimensions": list(dict.fromkeys(dimensions)),
            }
            time_grain = str(raw_contract.get("time_grain") or "").strip().lower()
            if time_grain:
                if time_grain not in {"day", "week", "month", "quarter", "year"}:
                    raise ValueError(
                        f"Queryability contract {contract_id!r} has unsupported time_grain {time_grain!r}."
                    )
                contract["time_grain"] = time_grain
            normalized.append(contract)
        return normalized

    def _handle_finalized_plan_call(self, sql_entries: List[SqlModelingEntry]) -> FuncToolResult:
        """Keep a successful plan stable when the model calls the tool again."""
        assert self._plan is not None
        try:
            entries = [SqlModelingEntry.model_validate(item) for item in sql_entries or []]
        except Exception as exc:
            return FuncToolResult(
                success=0,
                error=f"The SQL modeling plan is already finalized; invalid additional entries were ignored: {exc}",
                result={
                    "status": str(self._plan.candidate_plan.get("planning_status") or "ready"),
                    **self._plan.prompt_payload(),
                },
            )

        unchanged = all(
            entry.source_index in self._pending_entries
            and self._pending_entries[entry.source_index].model_dump() == entry.model_dump()
            for entry in entries
        )
        if unchanged:
            status = str(self._plan.candidate_plan.get("planning_status") or "ready")
            return FuncToolResult(result={"status": status, **self._plan.prompt_payload()})
        return FuncToolResult(
            success=0,
            error="The SQL modeling plan is already finalized; additional or changed entries were ignored.",
            result={
                "status": str(self._plan.candidate_plan.get("planning_status") or "ready"),
                **self._plan.prompt_payload(),
            },
        )

    def _validate_entries(self, entries: list[SqlModelingEntry]) -> str:
        indexes: set[int] = set()

        for index, entry in enumerate(entries, 1):
            if not entry.sql.strip():
                return f"sql_entries[{index - 1}].sql must not be empty."

            if entry.source_index in indexes:
                return f"Duplicate SQL source_index: {entry.source_index}."
            indexes.add(entry.source_index)

        return ""


def _normalize_business_name(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "").strip())
    return re.sub(r"_+", "_", text).strip("_").lower()


def _metric_output_ids(candidate_plan: Dict[str, Any]) -> list[str]:
    output_ids = []
    seen = set()
    for output in candidate_plan.get("outputs") or []:
        if not isinstance(output, dict):
            continue
        output_id = str(output.get("output_id") or "").strip()
        if output_id and output_id not in seen:
            seen.add(output_id)
            output_ids.append(output_id)
    return output_ids


def _queryability_contract_ids(candidate_plan: Dict[str, Any]) -> list[str]:
    contract_ids = []
    seen = set()
    for contract in candidate_plan.get("queryability_contracts") or []:
        if not isinstance(contract, dict):
            continue
        contract_id = str(contract.get("contract_id") or "").strip()
        if contract_id and contract_id not in seen:
            seen.add(contract_id)
            contract_ids.append(contract_id)
    return contract_ids


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
        # Catalog and schema discovery remain available as live tools. Keeping
        # their snapshots out of the authoring plan prevents stale suggestions
        # from constraining later LLM corrections.
        metric_catalog = list(existing_metric_catalog or [])
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
                "outputs": [],
            }
        analysis = dict(result.result or {})
        parse_errors = [item for item in analysis.get("parse_errors") or [] if isinstance(item, dict)]
        failed_sources = [
            str(item.get("source") or item.get("source_sql_name") or "<unknown>") for item in parse_errors
        ]
        if parse_errors and len(set(failed_sources)) >= len(sources):
            return {
                "available": False,
                "outputs": [],
                "parse_errors": parse_errors,
                "error": (
                    "SQL modeling analysis could not parse any submitted statement. "
                    f"Unresolved sources: {', '.join(failed_sources)}."
                ),
            }

        outputs = []
        for contract in analysis.get("output_contracts") or []:
            if not isinstance(contract, dict):
                continue
            output_id = str(contract.get("output_id") or "").strip()
            if not output_id:
                continue
            outputs.append(
                {
                    "output_id": output_id,
                    "source_id": str(contract.get("source_sql_name") or "").strip(),
                    "name": str(contract.get("output_name") or "").strip(),
                    "expression": str(contract.get("expression") or "").strip(),
                    "role": str(contract.get("output_role") or "metric").strip().lower(),
                }
            )

        queryability_contracts = []
        contract_counts: Dict[str, int] = {}
        for raw_contract in analysis.get("queryability_contracts") or []:
            if not isinstance(raw_contract, dict):
                continue
            source_id = str(raw_contract.get("source") or "").strip()
            dimensions = [
                str(dimension).strip()
                for dimension in raw_contract.get("dimension_hints") or []
                if str(dimension).strip()
            ]
            metric_output_ids = [
                str(output_id).strip()
                for output_id in raw_contract.get("metric_output_ids") or []
                if str(output_id).strip()
            ]
            if not source_id or not dimensions or not metric_output_ids:
                continue
            contract_counts[source_id] = contract_counts.get(source_id, 0) + 1
            contract: Dict[str, Any] = {
                "contract_id": f"{source_id}:group_{contract_counts[source_id]}",
                "source_id": source_id,
                "metric_output_ids": list(dict.fromkeys(metric_output_ids)),
                "dimensions": list(dict.fromkeys(dimensions)),
            }
            grains = list(
                dict.fromkeys(
                    str(hint.get("grain") or "").strip().lower()
                    for hint in raw_contract.get("time_group_hints") or []
                    if isinstance(hint, dict) and str(hint.get("grain") or "").strip()
                )
            )
            if not grains:
                grains = list(
                    dict.fromkeys(
                        match.group(1).lower()
                        for dimension in dimensions
                        for match in [re.search(r"(?:^|_)(day|week|month|quarter|year)(?:$|_)", dimension, re.I)]
                        if match is not None
                    )
                )
            if len(grains) == 1:
                contract["time_grain"] = grains[0]
            queryability_contracts.append(contract)

        plan = {
            "available": True,
            "planning_status": "partial" if parse_errors else "ready",
            "outputs": outputs,
            "queryability_contracts": queryability_contracts,
            "summary": f"Found {len(outputs)} final SQL outputs from {len(sources)} source queries",
        }
        if parse_errors:
            plan["parse_errors"] = parse_errors
            plan["unresolved_sources"] = [
                {
                    "source_sql_name": str(item.get("source") or item.get("source_sql_name") or "<unknown>"),
                    "status": "blocked",
                    "reason": str(item.get("error") or "SQL parsing failed"),
                }
                for item in parse_errors
            ]
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
