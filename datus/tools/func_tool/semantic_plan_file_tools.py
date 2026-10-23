# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Store an optional semantic plan as a deterministic, reviewable graph."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from datus.storage.semantic_model.artifact_file import atomic_write_text, path_mutation_lock
from datus.tools.func_tool.base import FuncToolResult, tool_schema

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows keeps the in-process lock
    fcntl = None


@contextmanager
def _plan_mutation_lock(target: Path):
    """Keep baseline comparison and replacement together across API workers."""
    with path_mutation_lock(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_name(f".{target.name}.lock").open("a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _reject_non_finite_number(value: str):
    raise ValueError(f"{value} is not valid JSON")


_LAYERS = [
    {"index": 0, "kind": "physical_table", "label": "physical table"},
    {"index": 1, "kind": "dataset", "label": "dataset"},
    {"index": 2, "kind": "metric_atomic", "label": "atomic metric"},
    {"index": 3, "kind": "metric_derived", "label": "derived metric"},
]


def _plan_objects(plan: dict, key: str) -> list[dict]:
    items = plan.get(key, [])
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"{key} must be an array of objects")
    return items


def _plan_name(item: dict, key: str, context: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires a non-empty {key}")
    return value


def _plan_names(items: list[dict], key: str, context: str) -> set[str]:
    names = [_plan_name(item, key, context) for item in items]
    if len(names) != len(set(names)):
        raise ValueError(f"{context} names must be unique")
    return set(names)


def _string_list(item: dict, key: str, context: str) -> list[str]:
    values = item.get(key, [])
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{context}.{key} must be an array of non-empty strings")
    return values


def _metric_node_kind(metric: dict) -> str:
    kind = metric.get("kind")
    if kind == "atomic" or (kind in (None, "parameterized") and not metric.get("depends_on")):
        return "metric_atomic"
    return "metric_derived"


def _metric_edge_kind(metric: dict, ordinal: int) -> str:
    if metric.get("kind") == "filter":
        return "derive_base"
    if metric.get("kind") == "compose":
        return "compose_member"
    if metric.get("kind") == "window":
        return "window_base" if ordinal == 0 else "window_second"
    return "depends_on"


def _plan_graph(plan: dict, model_name: str, plan_id: str, revision: int) -> dict:
    """Project planning intent, not compiled lineage, into Dosi's four graph lanes."""
    unexpected = set(plan) - {"summary", "datasets", "field_decisions", "relationships", "metrics"}
    if unexpected:
        raise ValueError(f"Unexpected plan fields: {', '.join(sorted(unexpected))}")
    if not isinstance(plan.get("summary", ""), str):
        raise ValueError("summary must be a string")
    datasets = _plan_objects(plan, "datasets")
    field_decisions = _plan_objects(plan, "field_decisions")
    relationships = _plan_objects(plan, "relationships")
    metrics = _plan_objects(plan, "metrics")
    dataset_names = _plan_names(datasets, "name", "Dataset")
    _plan_names(relationships, "name", "Relationship")
    metric_names = _plan_names(metrics, "name", "Metric")

    fields_by_dataset: dict[str, list[dict]] = {}
    for decision in field_decisions:
        dataset = _plan_name(decision, "dataset", "Field decision")
        field = _plan_name(decision, "field", "Field decision")
        if dataset not in dataset_names:
            raise ValueError(f"Field decision {dataset}.{field} references an unplanned dataset")
        fields = fields_by_dataset.setdefault(dataset, [])
        if any(existing["field"] == field for existing in fields):
            raise ValueError(f"Duplicate field decision: {dataset}.{field}")
        fields.append(
            {"field": field, **{key: value for key, value in decision.items() if key not in {"dataset", "field"}}}
        )

    table_datasets: dict[str, list[str]] = {}
    dataset_sources: dict[str, tuple[str | None, list[str]]] = {}
    for dataset in datasets:
        name = dataset["name"]
        source_kind = dataset.get("source_kind")
        if source_kind not in (None, "physical", "query"):
            raise ValueError(f"Dataset {name} has an unknown source_kind")
        source = dataset.get("source")
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError(f"Dataset {name}.source must be a non-empty string")
        if source_kind == "query" and source:
            raise ValueError(f"Query-backed dataset {name} must not put SQL in source")
        if source_kind is None and source:
            raise ValueError(f"Dataset {name} requires source_kind when source is set")
        sources = [source] if source_kind == "physical" and source else _string_list(dataset, "source_tables", name)
        if dataset.get("action") in {"create", "update"} and source_kind in {"physical", "query"} and not sources:
            raise ValueError(f"Dataset {name} requires a physical source or query source_tables")
        if len(sources) != len(set(sources)):
            raise ValueError(f"Dataset {name} contains duplicate source tables")
        dataset_sources[name] = (source_kind, sources)
        for table in sources:
            table_datasets.setdefault(table, []).append(name)

    nodes = [
        {
            "id": f"table:{table}",
            "kind": "physical_table",
            "layer": 0,
            "name": table,
            "label": table,
            "detail": {"table": table, "parts": table.split("."), "datasets": consumers},
        }
        for table, consumers in table_datasets.items()
    ]
    edges = []
    for dataset in datasets:
        name = dataset["name"]
        source_kind, sources = dataset_sources[name]
        nodes.append(
            {
                "id": f"dataset:{name}",
                "kind": "dataset",
                "layer": 1,
                "name": name,
                "label": name,
                "action": dataset.get("action"),
                "detail": {
                    **{key: value for key, value in dataset.items() if key not in {"name", "action", "source_kind"}},
                    "source_kind": "table" if source_kind == "physical" else source_kind,
                    "source_tables": sources,
                    "fields": fields_by_dataset.get(name, []),
                },
            }
        )
        for table in sources:
            edges.append(
                {
                    "id": f"reads_table:table:{table}->dataset:{name}",
                    "kind": "reads_table",
                    "source": f"table:{table}",
                    "target": f"dataset:{name}",
                    "via": "source" if source_kind == "physical" else "source_sql",
                }
            )

    for metric in metrics:
        name = metric["name"]
        dataset = metric.get("dataset")
        if dataset is not None and dataset not in dataset_names:
            raise ValueError(f"Metric {name} references an unplanned dataset: {dataset}")
        dependencies = _string_list(metric, "depends_on", name)
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"Metric {name} has duplicate dependencies")
        for dependency in dependencies:
            if dependency not in metric_names:
                raise ValueError(f"Metric {name} references an unplanned metric: {dependency}")
        kind = _metric_node_kind(metric)
        nodes.append(
            {
                "id": f"metric:{name}",
                "kind": kind,
                "layer": 2 if kind == "metric_atomic" else 3,
                "name": name,
                "label": name,
                "action": metric.get("action"),
                "detail": {
                    **{key: value for key, value in metric.items() if key not in {"name", "action", "kind"}},
                    "metric_kind": metric.get("kind") or ("atomic" if kind == "metric_atomic" else "derived"),
                },
            }
        )
        if dataset:
            edges.append(
                {
                    "id": f"aggregates:dataset:{dataset}->metric:{name}",
                    "kind": "aggregates",
                    "source": f"dataset:{dataset}",
                    "target": f"metric:{name}",
                }
            )
        for ordinal, dependency in enumerate(dependencies):
            edge_kind = _metric_edge_kind(metric, ordinal)
            edge = {
                "id": f"{edge_kind}:metric:{dependency}->metric:{name}#{ordinal}",
                "kind": edge_kind,
                "source": f"metric:{dependency}",
                "target": f"metric:{name}",
            }
            if edge_kind == "compose_member":
                edge["ordinal"] = ordinal
            edges.append(edge)

    for relationship in relationships:
        name = relationship["name"]
        source = _plan_name(relationship, "from_dataset", f"Relationship {name}")
        target = _plan_name(relationship, "to_dataset", f"Relationship {name}")
        if source not in dataset_names or target not in dataset_names:
            raise ValueError(f"Relationship {name} references an unplanned dataset")
        _string_list(relationship, "from_fields", name)
        _string_list(relationship, "to_fields", name)
        edges.append(
            {
                "id": f"join:{name}",
                "kind": "join",
                "source": f"dataset:{source}",
                "target": f"dataset:{target}",
                "name": name,
                "action": relationship.get("action", "reuse"),
                "detail": {
                    key: value
                    for key, value in relationship.items()
                    if key not in {"name", "action", "from_dataset", "to_dataset"}
                },
            }
        )

    return {
        "version": 1,
        "model": model_name,
        "plan": {"id": plan_id, "revision": revision, "summary": plan.get("summary", "")},
        "layers": _LAYERS,
        "nodes": nodes,
        "edges": edges,
    }


@dataclass(frozen=True)
class PreparedSemanticPlan:
    model_name: str
    path: str
    old_text: str | None
    new_text: str
    revision: int
    counts: dict[str, int]

    def preview(self) -> dict[str, Any]:
        return {"filePath": self.path, "oldText": self.old_text or "", "newText": self.new_text}


class SemanticPlanFileTools:
    """Store a reviewable plan graph without approving its design."""

    permission_category = "semantic_tools"
    _MODEL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self._plan_ids: dict[str, str] = {}
        self._revisions: dict[str, int] = {}
        self._write_lock = RLock()

    def available_tools(self):
        from datus.tools.func_tool import trans_to_function_tool

        return [trans_to_function_tool(self.write_semantic_model_plan_file)]

    @tool_schema(strict_mode=False)
    def write_semantic_model_plan_file(self, model_name: str, plan_json: dict[str, Any] | str) -> FuncToolResult:
        """Convert planning JSON to a graph and write it without changing the model.

        Optional human-review artifact, not a prerequisite for authoring or target
        selection. The plan uses the format below, not OSI dataset/metric objects.
        Saving checks its structure and references; it does not validate the model,
        approve a source choice or execute a query.

        Args:
            model_name: Selected semantic-model name; revisions reuse its plan directory.
            plan_json: Structured object; legacy JSON-encoded objects also work.
                Only these top-level keys are accepted (all optional):
                summary (string), datasets, field_decisions, relationships, metrics
                (arrays of objects, default empty). Each dataset, relationship and
                metric requires a unique nonempty name within its array.
                Dataset: name, optional action (create/update/reuse), source_kind
                (physical/query). For physical use source as the table name; for
                query use source_tables as an array of table names and omit source
                entirely (no SELECT in a plan). Create/update with physical/query
                source_kind requires sources.
                Record query rowset_semantics and native_gap or user_requirement.
                Field decision: dataset and field are required; dataset must name
                a planned dataset. Optional role, group_by and reason are retained.
                Relationship: name, from_dataset, to_dataset are required; both
                datasets must be planned. Optional from_fields/to_fields are arrays
                of field names in matching order; action and reason are retained.
                Metric: name, optional dataset (a planned dataset), kind
                (atomic/parameterized/filter/compose/window), depends_on (array of
                planned metric names, no duplicates), action, role and reason.
                Include reused datasets/metrics when other objects reference them.

        Returns:
            Workspace-relative JSON graph path, revision and object counts, or a
            validation/write error. The semantic model file is never changed.
        """
        # CLI and non-reviewed API calls still execute directly. The web review
        # adapter calls prepare/commit separately and never writes before Accept.
        with self._write_lock:
            prepared = self.prepare(model_name, plan_json)
            if isinstance(prepared, FuncToolResult):
                return prepared
            result = self.commit(prepared)
            if result.success:
                result.result["review_status"] = "applied"
            return result

    def _existing_plan_id(self, model_name: str) -> str | None:
        parent = self.project_root / ".datus/semantic-model-plans"
        if not parent.resolve().is_relative_to(self.project_root):
            return None
        candidates = []
        for path in parent.glob("*/semantic-model-plan.json"):
            if not path.resolve().is_relative_to(self.project_root):
                continue
            try:
                graph = json.loads(path.read_text(encoding="utf-8"))
                if graph.get("model") == model_name and graph.get("plan", {}).get("id") == path.parent.name:
                    candidates.append((path.stat().st_mtime_ns, path.parent.name))
            except (OSError, ValueError, AttributeError):
                continue
        return max(candidates)[1] if candidates else None

    @staticmethod
    def _model_plan_id(model_name: str) -> str:
        # The containing project directory provides project isolation. Keeping
        # the ID independent of absolute mount paths also coordinates workers
        # that mount the same project at different locations.
        return uuid5(NAMESPACE_URL, f"datus:semantic-model-plan:{model_name}").hex

    def prepare(self, model_name: str, plan_json: dict[str, Any] | str) -> PreparedSemanticPlan | FuncToolResult:
        """Validate and freeze a proposal without changing its on-disk baseline."""
        if not isinstance(model_name, str) or not self._MODEL_NAME.fullmatch(model_name):
            return FuncToolResult(success=0, error="model_name must be a simple semantic-model name")
        try:
            plan = (
                json.loads(plan_json, parse_constant=_reject_non_finite_number)
                if isinstance(plan_json, str)
                else plan_json
            )
            if not isinstance(plan, dict):
                return FuncToolResult(success=0, error="plan_json must be a JSON object")
            json.dumps(plan, allow_nan=False)
        except (TypeError, ValueError) as exc:
            return FuncToolResult(success=0, error=f"plan_json must be valid JSON: {exc}")

        with self._write_lock:
            plan_id = (
                self._existing_plan_id(model_name) or self._plan_ids.get(model_name) or self._model_plan_id(model_name)
            )
            relative_path = Path(".datus/semantic-model-plans") / plan_id / "semantic-model-plan.json"
            target = self.project_root / relative_path
            if not target.resolve(strict=False).is_relative_to(self.project_root):
                return FuncToolResult(success=0, error="Semantic plan path must stay inside the project workspace")
            try:
                old_text = target.read_text(encoding="utf-8") if target.exists() else None
                previous = json.loads(old_text).get("plan", {}).get("revision", 0) if old_text else 0
                revision = max(previous, self._revisions.get(model_name, 0)) + 1
                graph = _plan_graph(plan, model_name, plan_id, revision)
                content = json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            except (OSError, TypeError, ValueError, AttributeError) as exc:
                return FuncToolResult(success=0, error=f"Invalid semantic plan: {exc}")
            # Reserve the identity even if this proposal is rejected, so feedback
            # revises the same plan rather than minting another directory.
            self._plan_ids[model_name] = plan_id
            return PreparedSemanticPlan(
                model_name,
                relative_path.as_posix(),
                old_text,
                content,
                revision,
                {
                    "datasets": len(plan.get("datasets", [])),
                    "relationships": len(plan.get("relationships", [])),
                    "metrics": len(plan.get("metrics", [])),
                },
            )

    def commit(self, prepared: PreparedSemanticPlan) -> FuncToolResult:
        """Commit exactly the reviewed content, rejecting a stale baseline."""
        target = self.project_root / prepared.path
        with self._write_lock:
            if not target.resolve(strict=False).is_relative_to(self.project_root):
                return FuncToolResult(success=0, error="Semantic plan path must stay inside the project workspace")
            try:
                model_lock = (
                    self.project_root
                    / ".datus/semantic-model-plans"
                    / f"{self._model_plan_id(prepared.model_name)}.model"
                )
                if not model_lock.resolve(strict=False).is_relative_to(self.project_root):
                    return FuncToolResult(success=0, error="Semantic plan path must stay inside the project workspace")
                with _plan_mutation_lock(model_lock), _plan_mutation_lock(target):
                    accepted_id = self._existing_plan_id(prepared.model_name)
                    if accepted_id is not None and accepted_id != target.parent.name:
                        return FuncToolResult(
                            success=0, error="Plan changed during review. Submit a fresh proposal for review."
                        )
                    current = target.read_text(encoding="utf-8") if target.exists() else None
                    if current != prepared.old_text:
                        return FuncToolResult(
                            success=0, error="Plan changed during review. Submit a fresh proposal for review."
                        )
                    atomic_write_text(target, prepared.new_text)
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Could not write semantic plan: {exc}")
            self._revisions[prepared.model_name] = prepared.revision
            return FuncToolResult(
                result={"path": prepared.path, "revision": prepared.revision, "counts": prepared.counts}
            )
