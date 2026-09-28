# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Store an optional semantic plan as a deterministic, reviewable graph."""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from uuid import uuid4

from datus.storage.semantic_model.artifact_file import atomic_write_text, path_mutation_lock
from datus.tools.func_tool.base import FuncToolResult


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


class SemanticPlanFileTools:
    """Store a reviewable plan graph without approving its design."""

    permission_category = "semantic_tools"
    _MODEL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self._plan_ids: dict[str, str] = {}
        self._revisions: dict[str, int] = {}
        self._write_lock = Lock()

    def available_tools(self):
        from datus.tools.func_tool import trans_to_function_tool

        return [trans_to_function_tool(self.write_semantic_model_plan_file)]

    def write_semantic_model_plan_file(self, model_name: str, plan_json: str) -> FuncToolResult:
        """Convert planning JSON to a graph and write it without changing the model.

        Use with a planning workflow that prepares the JSON for human review.

        Args:
            model_name: Selected semantic-model name; revisions reuse its plan directory.
            plan_json: Flat JSON object following the loaded planning skill's framework.
        """
        if not isinstance(model_name, str) or not self._MODEL_NAME.fullmatch(model_name):
            return FuncToolResult(success=0, error="model_name must be a simple semantic-model name")
        try:
            plan = json.loads(plan_json, parse_constant=_reject_non_finite_number)
        except (TypeError, ValueError) as exc:
            return FuncToolResult(success=0, error=f"plan_json must be valid JSON: {exc}")
        if not isinstance(plan, dict):
            return FuncToolResult(success=0, error="plan_json must be a JSON object")
        try:
            json.dumps(plan, allow_nan=False)
        except (TypeError, ValueError) as exc:
            return FuncToolResult(success=0, error=f"plan_json must be valid JSON: {exc}")
        with self._write_lock:
            plan_id = self._plan_ids.get(model_name)
            if plan_id is None:
                plan_id = uuid4().hex
            revision = self._revisions.get(model_name, 0) + 1
            try:
                graph = _plan_graph(plan, model_name, plan_id, revision)
                content = json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            except (TypeError, ValueError) as exc:
                return FuncToolResult(success=0, error=f"Invalid semantic plan: {exc}")
            relative_path = Path(".datus/semantic-model-plans") / plan_id / "semantic-model-plan.json"
            target = self.project_root / relative_path
            if not target.resolve(strict=False).is_relative_to(self.project_root):
                return FuncToolResult(success=0, error="Semantic plan path must stay inside the project workspace")
            try:
                with path_mutation_lock(target):
                    atomic_write_text(target, content)
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Could not write semantic plan: {exc}")
            self._plan_ids[model_name] = plan_id
            self._revisions[model_name] = revision
            return FuncToolResult(
                result={
                    "path": relative_path.as_posix(),
                    "revision": revision,
                    "counts": {
                        "datasets": len(plan.get("datasets", [])),
                        "relationships": len(plan.get("relationships", [])),
                        "metrics": len(plan.get("metrics", [])),
                    },
                }
            )
