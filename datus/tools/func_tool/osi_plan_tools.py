# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Intent planning and approval gate for unified OSI semantic authoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from datus.storage.semantic_model.artifact_file import atomic_write_text
from datus.tools.func_tool.base import FuncToolResult


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SemanticPlanDataset(_PlanModel):
    name: str = Field(min_length=1)
    action: Literal["reuse", "create", "update", "delete"]
    source_kind: Optional[Literal["physical", "query"]] = None
    source: Optional[str] = Field(
        default=None,
        description="Qualified table for a physical dataset. Omit implementation SQL for query-backed plans.",
    )
    source_tables: List[str] = Field(
        default_factory=list,
        description="Qualified physical tables read by a query-backed dataset; omit SQL text.",
    )
    grain: str = Field(default="", description="One-row grain of this dataset.")
    reason: str = Field(default="", description="Why this dataset is needed.")
    rowset_semantics: str = Field(default="", description="Exact row-level transformation for query-backed data.")
    native_gap: str = Field(default="", description="Why physical datasets plus native metrics cannot express it.")


class SemanticPlanFieldDecision(_PlanModel):
    dataset: str = Field(min_length=1)
    field: str = Field(min_length=1)
    role: Literal["dimension", "measure", "identifier", "time", "attribute"]
    group_by: Optional[bool] = None
    additive: Optional[bool] = None
    non_additive_over: List[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class SemanticPlanRelationship(_PlanModel):
    name: str = Field(min_length=1)
    action: Literal["reuse", "create", "update", "delete"] = "reuse"
    from_dataset: str = Field(default="", description="Source dataset node name.")
    to_dataset: str = Field(default="", description="Target dataset node name.")
    from_fields: List[str] = Field(default_factory=list)
    to_fields: List[str] = Field(default_factory=list)
    cardinality: str = ""
    reason: str = Field(default="", description="Why the relationship is needed.")


class SemanticPlanMetric(_PlanModel):
    name: str = Field(min_length=1)
    action: Literal["reuse", "create", "update", "delete"]
    kind: Optional[Literal["atomic", "filter", "compose", "window", "parameterized"]] = None
    role: Optional[Literal["business_output", "helper"]] = None
    dataset: Optional[str] = None
    grain: str = ""
    definition: str = Field(default="", description="Concise business definition, not necessarily engine syntax.")
    depends_on: List[str] = Field(default_factory=list)
    additivity: Literal["additive", "semi_additive", "non_additive", "recompute", "infer"] = "infer"
    non_additive_over: List[str] = Field(default_factory=list)


class OsiSemanticModelPlan(_PlanModel):
    """Structured authoring intent submitted before any semantic-model mutation."""

    summary: str = Field(min_length=1)
    datasets: List[SemanticPlanDataset] = Field(default_factory=list)
    field_decisions: List[SemanticPlanFieldDecision] = Field(default_factory=list)
    relationships: List[SemanticPlanRelationship] = Field(default_factory=list)
    metrics: List[SemanticPlanMetric] = Field(default_factory=list)


class OsiSemanticModelPlanState:
    """Request-local semantic intent and approval state."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        execution_mode: Literal["interactive", "workflow"],
        authoring_scope: Literal["datasets", "full"],
    ):
        self.project_root = Path(project_root).expanduser().resolve(strict=False)
        self.execution_mode = execution_mode
        self.authoring_scope = authoring_scope
        self.reset()

    def reset(self) -> None:
        self.plan: Optional[OsiSemanticModelPlan] = None
        self.plan_id = ""
        self.revision = 0
        self.status: Literal["missing", "pending_confirmation", "approved"] = "missing"
        self.target: Dict[str, Any] = {}
        self.plan_file = ""

    @staticmethod
    def _normalize(value: Any) -> str:
        return str(value or "").strip().casefold()

    def _artifact_dir(self) -> Path:
        return self.project_root / ".datus" / "semantic-model-plans" / self.plan_id

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _compact(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Drop absent optional detail while preserving meaningful false values."""
        return {
            key: value
            for key, value in payload.items()
            if value is not None and value != "" and value != [] and value != {}
        }

    @staticmethod
    def _metric_node_kind(metric: SemanticPlanMetric) -> str:
        if metric.kind == "atomic" or (metric.kind == "parameterized" and not metric.depends_on):
            return "metric_atomic"
        return "metric_derived"

    @staticmethod
    def _metric_dependency_edge_kind(metric: SemanticPlanMetric, ordinal: int) -> str:
        if metric.kind == "filter":
            return "derive_base"
        if metric.kind == "compose":
            return "compose_member"
        if metric.kind == "window":
            return "window_base" if ordinal == 0 else "window_second"
        return "depends_on"

    def _plan_graph(self) -> Dict[str, Any]:
        """Project concise authoring intent into the same lane/node/edge shape as Dosi lineage."""
        if self.plan is None:
            return {}

        plan = self.plan
        fields_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
        for decision in plan.field_decisions:
            detail = decision.model_dump(mode="json", exclude={"dataset"}, exclude_none=True)
            fields_by_dataset.setdefault(decision.dataset, []).append(self._compact(detail))

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        physical_tables: List[str] = []
        for dataset in plan.datasets:
            sources = (
                [dataset.source] if dataset.source_kind == "physical" and dataset.source else dataset.source_tables
            )
            for table in sources:
                if table not in physical_tables:
                    physical_tables.append(table)

        for table in physical_tables:
            nodes.append(
                {
                    "id": f"table:{table}",
                    "kind": "physical_table",
                    "layer": 0,
                    "name": table,
                }
            )

        for dataset in plan.datasets:
            detail = self._compact(
                {
                    "source_kind": "table" if dataset.source_kind == "physical" else dataset.source_kind,
                    "source": dataset.source if dataset.source_kind == "physical" else None,
                    "source_tables": (
                        [dataset.source]
                        if dataset.source_kind == "physical" and dataset.source
                        else dataset.source_tables
                    ),
                    "grain": dataset.grain,
                    "reason": dataset.reason,
                    "rowset_semantics": dataset.rowset_semantics,
                    "native_gap": dataset.native_gap,
                    "fields": fields_by_dataset.get(dataset.name, []),
                }
            )
            nodes.append(
                {
                    "id": f"dataset:{dataset.name}",
                    "kind": "dataset",
                    "layer": 1,
                    "name": dataset.name,
                    "action": dataset.action,
                    "detail": detail,
                }
            )
            source_tables = (
                [dataset.source] if dataset.source_kind == "physical" and dataset.source else dataset.source_tables
            )
            for table in source_tables:
                source = f"table:{table}"
                target = f"dataset:{dataset.name}"
                edges.append(
                    {
                        "id": f"reads_table:{source}->{target}",
                        "kind": "reads_table",
                        "source": source,
                        "target": target,
                        "via": "source" if dataset.source_kind == "physical" else "source_sql",
                    }
                )

        for metric in plan.metrics:
            node_kind = self._metric_node_kind(metric)
            nodes.append(
                {
                    "id": f"metric:{metric.name}",
                    "kind": node_kind,
                    "layer": 2 if node_kind == "metric_atomic" else 3,
                    "name": metric.name,
                    "action": metric.action,
                    "detail": self._compact(
                        {
                            "metric_kind": metric.kind,
                            "role": metric.role,
                            "grain": metric.grain,
                            "definition": metric.definition,
                            "additivity": metric.additivity,
                            "non_additive_over": metric.non_additive_over,
                        }
                    ),
                }
            )

        for relationship in plan.relationships:
            edges.append(
                {
                    "id": f"join:{relationship.name}",
                    "kind": "join",
                    "source": f"dataset:{relationship.from_dataset}",
                    "target": f"dataset:{relationship.to_dataset}",
                    "action": relationship.action,
                    "detail": self._compact(
                        {
                            "from_fields": relationship.from_fields,
                            "to_fields": relationship.to_fields,
                            "cardinality": relationship.cardinality,
                            "reason": relationship.reason,
                        }
                    ),
                }
            )

        for metric in plan.metrics:
            target = f"metric:{metric.name}"
            if metric.dataset:
                source = f"dataset:{metric.dataset}"
                edges.append(
                    {
                        "id": f"aggregates:{source}->{target}",
                        "kind": "aggregates",
                        "source": source,
                        "target": target,
                    }
                )
            for ordinal, dependency in enumerate(metric.depends_on):
                source = f"metric:{dependency}"
                kind = self._metric_dependency_edge_kind(metric, ordinal)
                suffix = f"#{ordinal}" if kind == "compose_member" else ""
                edge = {
                    "id": f"{kind}:{source}->{target}{suffix}",
                    "kind": kind,
                    "source": source,
                    "target": target,
                }
                if kind == "compose_member":
                    edge["ordinal"] = ordinal
                edges.append(edge)

        return {
            "version": 1,
            "model": str(self.target.get("semantic_model_name") or ""),
            "plan": self._compact(
                {
                    "id": self.plan_id,
                    "revision": self.revision,
                    "status": self.status,
                    "summary": plan.summary,
                    "model_file": self.target.get("semantic_model_file"),
                }
            ),
            "layers": [
                {"index": 0, "kind": "physical_table", "label": "physical table"},
                {"index": 1, "kind": "dataset", "label": "dataset"},
                {"index": 2, "kind": "metric_atomic", "label": "atomic metric"},
                {"index": 3, "kind": "metric_derived", "label": "derived metric"},
            ],
            "nodes": nodes,
            "edges": edges,
        }

    def _write_plan_artifact(self) -> None:
        path = self._artifact_dir() / "semantic-model-plan.json"
        self._write_json(path, self._plan_graph())
        self.plan_file = str(path)

    def _validate_intent(self, plan: OsiSemanticModelPlan) -> None:
        if not any((plan.datasets, plan.relationships, plan.metrics)):
            raise ValueError("The semantic-model plan must include at least one dataset, relationship, or metric.")

        for dataset in plan.datasets:
            if dataset.action in {"create", "update"} and not dataset.source_kind:
                raise ValueError(f"Dataset {dataset.name!r} requires source_kind for {dataset.action}.")
            if dataset.source_kind == "physical" and dataset.action in {"create", "update"} and not dataset.source:
                raise ValueError(f"Physical dataset {dataset.name!r} requires a qualified table source.")
            if dataset.source_kind == "query" and dataset.source:
                raise ValueError(
                    f"Query-backed dataset {dataset.name!r} must omit implementation SQL from the concise plan; "
                    "use rowset_semantics and native_gap."
                )
            if dataset.source_kind == "query" and dataset.action != "delete":
                missing = [
                    label
                    for label, value in (
                        ("source_tables", dataset.source_tables),
                        ("rowset_semantics", dataset.rowset_semantics),
                        ("native_gap", dataset.native_gap),
                    )
                    if not value
                ]
                if missing:
                    raise ValueError(f"Query-backed dataset {dataset.name!r} requires " + ", ".join(missing) + ".")
            if len(dataset.source_tables) != len(set(dataset.source_tables)):
                raise ValueError(f"Dataset {dataset.name!r} contains duplicate source_tables.")

        for relationship in plan.relationships:
            if not relationship.from_dataset or not relationship.to_dataset:
                raise ValueError(f"Relationship {relationship.name!r} requires from_dataset and to_dataset.")
            if relationship.action in {"create", "update"} and (
                not relationship.from_fields or not relationship.to_fields
            ):
                raise ValueError(f"Relationship {relationship.name!r} requires from_fields and to_fields.")

        for metric in plan.metrics:
            if metric.action in {"create", "update"} and (not metric.kind or not metric.role or not metric.definition):
                raise ValueError(
                    f"Metric {metric.name!r} requires kind, role, and a business definition for {metric.action}."
                )
            if metric.kind == "atomic" and metric.depends_on:
                raise ValueError(f"Atomic metric {metric.name!r} cannot depend on other metrics.")
            if len(metric.depends_on) != len(set(metric.depends_on)):
                raise ValueError(f"Metric {metric.name!r} contains duplicate dependencies.")

        dataset_names = [dataset.name for dataset in plan.datasets]
        relationship_names = [relationship.name for relationship in plan.relationships]
        metric_names = [metric.name for metric in plan.metrics]
        for label, names in (
            ("dataset", dataset_names),
            ("relationship", relationship_names),
            ("metric", metric_names),
        ):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"The plan contains duplicate {label} names: {', '.join(duplicates)}.")

        dataset_name_set = set(dataset_names)
        metric_name_set = set(metric_names)
        field_refs = [(decision.dataset, decision.field) for decision in plan.field_decisions]
        duplicate_field_refs = sorted(
            {f"{dataset}.{field}" for dataset, field in field_refs if field_refs.count((dataset, field)) > 1}
        )
        if duplicate_field_refs:
            raise ValueError("The plan contains duplicate field decisions: " + ", ".join(duplicate_field_refs) + ".")
        for decision in plan.field_decisions:
            if decision.dataset not in dataset_name_set:
                raise ValueError(f"Field decision {decision.dataset}.{decision.field} references an unplanned dataset.")
        for relationship in plan.relationships:
            missing_datasets = [
                name for name in (relationship.from_dataset, relationship.to_dataset) if name not in dataset_name_set
            ]
            if missing_datasets:
                raise ValueError(
                    f"Relationship {relationship.name!r} references unplanned datasets: "
                    + ", ".join(missing_datasets)
                    + "."
                )
        for metric in plan.metrics:
            if metric.dataset and metric.dataset not in dataset_name_set:
                raise ValueError(f"Metric {metric.name!r} references unplanned dataset {metric.dataset!r}.")
            missing_dependencies = [name for name in metric.depends_on if name not in metric_name_set]
            if missing_dependencies:
                raise ValueError(
                    f"Metric {metric.name!r} references unplanned metrics: " + ", ".join(missing_dependencies) + "."
                )

        if self.authoring_scope == "datasets":
            changed_metrics = [metric.name for metric in plan.metrics if metric.action != "reuse"]
            if changed_metrics:
                raise ValueError(
                    "Datasets-only semantic_modeling cannot plan metric mutations: " + ", ".join(changed_metrics)
                )

    def submit(
        self,
        plan: OsiSemanticModelPlan,
        *,
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not target:
            raise ValueError("Select one semantic-model target before submitting its authoring plan.")
        if self.target and (
            self._normalize(self.target.get("absolute_path")) != self._normalize(target.get("absolute_path"))
        ):
            raise ValueError("The semantic-model plan cannot switch to another selected target.")

        self._validate_intent(plan)
        if not self.plan_id:
            self.plan_id = uuid4().hex
        self.revision += 1
        self.plan = plan
        self.target = dict(target)
        self.status = "approved" if self.execution_mode == "workflow" else "pending_confirmation"
        self._write_plan_artifact()
        return self.public_summary()

    def approve_after_user_confirmation(self, plan_id: str) -> Dict[str, Any]:
        """Approve an interactive plan after the host has collected the user's decision."""
        if self.execution_mode != "interactive":
            raise ValueError("Workflow plans are approved automatically when submitted.")
        if not self.plan or not self.plan_id:
            raise ValueError("Submit a semantic-model plan before approving it.")
        if self.status != "pending_confirmation":
            raise ValueError("The semantic-model plan is not awaiting user confirmation.")
        if plan_id.strip() != self.plan_id:
            raise ValueError("plan_id does not match the pending semantic-model plan.")
        self.status = "approved"
        self._write_plan_artifact()
        return self.public_summary()

    def public_summary(self) -> Dict[str, Any]:
        plan = self.plan
        return {
            "plan_id": self.plan_id,
            "revision": self.revision,
            "status": self.status,
            "confirmation_required": self.status == "pending_confirmation",
            "plan_file": self.plan_file,
            "counts": {
                "datasets": len(plan.datasets) if plan else 0,
                "relationships": len(plan.relationships) if plan else 0,
                "metrics": len(plan.metrics) if plan else 0,
            },
            "graph": self._plan_graph(),
        }

    def require_approved(self, path: str | Path) -> None:
        if self.status != "approved" or self.plan is None:
            raise ValueError("Submit and approve the semantic-model authoring plan before modifying the selected YAML.")
        requested = Path(path).expanduser().resolve(strict=False)
        expected = Path(str(self.target.get("absolute_path") or "")).expanduser().resolve(strict=False)
        if requested != expected:
            raise ValueError("The approved semantic-model plan belongs to a different target path.")


class OsiSemanticModelPlanTools:
    """Tools that submit structured semantic-model authoring intent."""

    permission_category = "semantic_tools"

    def __init__(self, *, plan_state: OsiSemanticModelPlanState, target_state: Any):
        self.plan_state = plan_state
        self.target_state = target_state

    def submit_osi_semantic_model_plan(
        self,
        plan: OsiSemanticModelPlan,
    ) -> FuncToolResult:
        """Submit or replace the concise authoring plan before changing semantic YAML."""
        try:
            parsed = plan if isinstance(plan, OsiSemanticModelPlan) else OsiSemanticModelPlan.model_validate(plan)
            result = self.plan_state.submit(
                parsed,
                target=self.target_state.selected or {},
            )
            return FuncToolResult(result=result)
        except (ValidationError, ValueError) as exc:
            return FuncToolResult(
                success=0,
                error=f"Invalid semantic-model authoring plan: {exc}",
                result={"code": "semantic_model_plan_invalid"},
            )
