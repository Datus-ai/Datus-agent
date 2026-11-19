# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Store an optional semantic plan as a deterministic, reviewable graph."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from datus.storage.semantic_model.artifact_file import (
    atomic_write_text,
    ontology_bundle_lock,
    ontology_dir_for_model,
    path_mutation_lock,
    semantic_artifact_lock,
    semantic_model_load_lock,
)
from datus.tools.func_tool.base import FuncToolResult, tool_schema
from datus.tools.registry.contracts import tool_contract
from datus.tools.semantic_tools.exceptions import SemanticCoreException

if TYPE_CHECKING:
    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState

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


def _plan_action(item: dict, context: str) -> str | None:
    action = item.get("action")
    if action not in (None, "create", "reuse", "update", "delete"):
        raise ValueError(f"{context} has an unknown action: {action!r}")
    return action


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


def _plan_graph(plan: dict, model_name: str, plan_id: str, revision: int, baseline: dict | None = None) -> dict:
    """Project planning intent, not compiled lineage, into Dosi's four graph lanes."""
    unexpected = set(plan) - {"summary", "datasets", "field_decisions", "relationships", "metrics"}
    if unexpected:
        raise ValueError(f"Unexpected plan fields: {', '.join(sorted(unexpected))}")
    if not isinstance(plan.get("summary", ""), str):
        raise ValueError("summary must be a string")
    datasets = list(_plan_objects(plan, "datasets"))
    field_decisions = _plan_objects(plan, "field_decisions")
    relationships = _plan_objects(plan, "relationships")
    metrics = _plan_objects(plan, "metrics")
    dataset_names = _plan_names(datasets, "name", "Dataset")
    proposed_dataset_names = set(dataset_names)
    _plan_names(relationships, "name", "Relationship")
    metric_names = _plan_names(metrics, "name", "Metric")
    if baseline is not None:
        dataset_names.update(node["name"] for node in baseline["nodes"] if node["kind"] == "dataset")
        metric_names.update(
            node["name"] for node in baseline["nodes"] if node["kind"] in {"metric_atomic", "metric_derived"}
        )

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
    datasets.extend(
        {"name": name, "action": "update"} for name in fields_by_dataset if name not in proposed_dataset_names
    )

    table_datasets: dict[str, list[str]] = {}
    dataset_sources: dict[str, tuple[str | None, list[str]]] = {}
    for dataset in datasets:
        name = dataset["name"]
        action = dataset.get("action") if baseline is None else _plan_action(dataset, f"Dataset {name}")
        if baseline is not None and action == "delete" and name in fields_by_dataset:
            raise ValueError(f"Cannot plan field decisions for dataset {name} planned for deletion")
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
        if action in {"create", "update"} and source_kind in {"physical", "query"} and not sources:
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
                "action": metric.get("action") if baseline is None else _plan_action(metric, f"Metric {name}"),
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
            suffix = f"#{ordinal}" if baseline is None or edge_kind == "compose_member" else ""
            edge = {
                "id": f"{edge_kind}:metric:{dependency}->metric:{name}{suffix}",
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
                "action": relationship.get("action", "reuse")
                if baseline is None
                else _plan_action(relationship, f"Relationship {name}"),
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


def _repair_model_inventory(model_text: bytes, model_name: str, model_file: str) -> dict:
    """Keep readable declarations visible when the selected model needs repair."""
    import yaml

    from datus.agent.node.semantic_authoring import _is_query_backed_dataset
    from datus.tools.semantic_tools.dosi.model import datus_payload
    from datus.utils.sql_utils import strip_sql_comments

    try:
        document = yaml.safe_load(model_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot read YAML for model repair from {model_file}: {exc}") from exc
    if not isinstance(document, dict) or document.get("name") != model_name or "datasets" not in document:
        raise ValueError(f"Cannot identify the model to repair in {model_file}")

    def named_objects(value: Any) -> list[dict]:
        if not isinstance(value, list):
            return []
        return [
            obj for obj in value if isinstance(obj, dict) and isinstance(obj.get("name"), str) and obj["name"].strip()
        ]

    datasets = named_objects(document.get("datasets"))
    dataset_names = {dataset["name"] for dataset in datasets}
    relationships = []
    unresolved_joins = []
    for relationship in named_objects(document.get("relationships")):
        source, target = relationship.get("from"), relationship.get("to")
        if not all(isinstance(name, str) and name in dataset_names for name in (source, target)):
            unresolved_joins.append(f"join:{relationship['name']}")
            continue
        relationships.append(
            {
                "name": relationship["name"],
                "from_dataset": source,
                "to_dataset": target,
                "from_fields": _known_column_names(relationship.get("from_columns")),
                "to_fields": _known_column_names(relationship.get("to_columns")),
                "action": "reuse",
            }
        )

    readable_datasets = []
    for dataset in datasets:
        entry = {"name": dataset["name"], "action": "reuse"}
        source = dataset.get("source")
        extensions = dataset.get("custom_extensions")
        if (
            "source_sql" in dataset
            or (isinstance(extensions, (list, dict)) and _is_query_backed_dataset(dataset))
            or (
                isinstance(source, str)
                and re.match(r"\s*(?:\(\s*)*(?:select|with)\s", strip_sql_comments(source), re.IGNORECASE)
            )
        ):
            entry["source_kind"] = "query"
        elif isinstance(source, str) and source.strip():
            entry.update(source_kind="physical", source=source)
        readable_datasets.append(entry)

    graph = _plan_graph(
        {
            "datasets": readable_datasets,
            "field_decisions": [
                {"dataset": dataset["name"], "field": field["name"]}
                for dataset in datasets
                for field in named_objects(dataset.get("fields"))
            ],
            "metrics": [
                {
                    "name": metric["name"],
                    "action": "reuse",
                    "definition": metric.get("description"),
                    "kind": "derived" if {"derive", "window"} & datus_payload(metric).keys() else "atomic",
                }
                for metric in named_objects(document.get("metrics"))
            ],
            "relationships": relationships,
        },
        model_name,
        "",
        0,
        {"nodes": []},
    )
    return {
        **{key: graph[key] for key in ("layers", "nodes", "edges")},
        "lineage_incomplete": True,
        "unresolved_joins": unresolved_joins,
    }


def _known_column_names(value: Any) -> list[str]:
    return [name for name in value if isinstance(name, str) and name.strip()] if isinstance(value, list) else []


def _model_graph(model_text: bytes | None, model_name: str, model_file: str, *, repairable: bool = False) -> dict:
    """Project the exact selected YAML into the existing plan wire format."""
    if model_text is None:
        return {"layers": _LAYERS, "nodes": [], "edges": []}

    from datus.tools.semantic_tools.dosi.engine import load_binding

    binding = load_binding()
    try:
        graph = binding.Engine(model_text=model_text.decode("utf-8")).lineage(redact_sql=True)
    except binding.ModelError as exc:
        if repairable:
            return _repair_model_inventory(model_text, model_name, model_file)
        raise ValueError(f"Cannot read model lineage from {model_file}: {exc}") from exc
    if graph["model"] != model_name:
        raise ValueError(f"Selected YAML declares {graph['model']!r}, expected {model_name!r}")

    nodes = []
    for item in graph["nodes"]:
        node = {key: item[key] for key in ("id", "kind", "layer", "name", "label")}
        detail = item["detail"]
        if item["kind"] == "dataset":
            tables = detail["source_tables"]
            node["detail"] = {
                "source_kind": detail["source_kind"],
                "source": tables[0] if detail["source_kind"] == "table" and tables else None,
                "source_tables": tables,
                "fields": [
                    {
                        "field": field["name"],
                        **({"reason": field["description"]} if field.get("description") else {}),
                    }
                    for field in detail.get("fields", [])
                ],
            }
            node["action"] = "reuse"
        elif item["kind"] in {"metric_atomic", "metric_derived"}:
            node["detail"] = {
                "definition": detail.get("description"),
                "metric_kind": "atomic" if item["kind"] == "metric_atomic" else detail["kind"],
            }
            node["action"] = "reuse"
        else:
            node["detail"] = detail
        nodes.append(node)
    edges = []
    for item in graph["edges"]:
        edge = {key: item[key] for key in ("id", "kind", "source", "target", "via", "ordinal") if key in item}
        if item["kind"] == "join":
            edge["name"] = item["id"].removeprefix("join:")
            pairs = item.get("column_pairs", [])
            edge["detail"] = {
                "from_fields": [pair[0] for pair in pairs],
                "to_fields": [pair[1] for pair in pairs],
                "cardinality": item.get("cardinality"),
                "join_kind": item.get("join_kind"),
            }
            edge["action"] = "reuse"
        edges.append(edge)
    return {"layers": _LAYERS, "nodes": nodes, "edges": edges}


def _merge_plan_graph(baseline: dict, proposal: dict) -> dict:
    """Apply this round's declarations to a copy of the fixed model graph."""
    nodes = {node["id"]: node for node in baseline["nodes"]}
    edges = {edge["id"]: edge for edge in baseline["edges"]}
    replaced_inputs: set[str] = set()
    deleted: set[str] = set()
    for node in proposal["nodes"]:
        node_id = node["id"]
        action = node.get("action")
        if node["kind"] == "physical_table":
            nodes.setdefault(node_id, node)
        elif action == "delete":
            if node_id not in nodes:
                raise ValueError(f"Cannot plan deletion of a missing object: {node_id}")
            deleted.add(node_id)
        elif (
            node_id not in nodes
            or action in {"create", "update"}
            or (node["kind"] == "dataset" and node["detail"]["fields"])
        ):
            if node_id in nodes and action == "create":
                raise ValueError(f"{node_id} already exists; mark it as reuse or update")
            if node_id not in nodes and action in {"reuse", "update"}:
                raise ValueError(f"Cannot {action} a missing object: {node_id}")
            if node_id in nodes and node["kind"] == "dataset":
                previous = nodes[node_id]["detail"]
                detail = (
                    {**previous, **node["detail"]}
                    if action == "update"
                    else {**previous, "fields": node["detail"]["fields"]}
                )
                if action != "update" or node["detail"]["source_kind"] is None:
                    for key in ("source", "source_kind", "source_tables"):
                        if key in previous:
                            detail[key] = previous[key]
                else:
                    if detail["source_kind"] == "query":
                        detail["source"] = None
                    if (detail["source_kind"], detail["source_tables"]) != (
                        previous["source_kind"],
                        previous["source_tables"],
                    ):
                        replaced_inputs.add(node_id)
                fields = {field["field"]: field for field in previous["fields"]}
                for field in detail["fields"]:
                    name = field["field"]
                    fields[name] = {**fields.get(name, {}), **field}
                detail["fields"] = list(fields.values())
                node = {**(node if action == "update" else nodes[node_id]), "detail": detail}
            else:
                replaced_inputs.add(node_id)
            nodes[node_id] = node

    proposed_joins = []
    known_joins = {edge_id for edge_id, edge in edges.items() if edge["kind"] == "join"}
    known_joins.update(baseline.get("unresolved_joins", ()))
    for edge in proposal["edges"]:
        if edge["kind"] == "join":
            edge_id = edge["id"]
            action = edge["action"] or ("reuse" if edge_id in known_joins else "create")
            if action == "create" and edge_id in known_joins:
                raise ValueError(f"Relationship {edge['name']} already exists; mark it as reuse or update")
            if action in {"update", "reuse", "delete"} and edge_id not in known_joins:
                raise ValueError(f"Cannot {action} a missing relationship: {edge['name']}")
            if action == "delete":
                edges.pop(edge_id, None)
            elif action == "reuse":
                existing = edges.get(edge_id, edge)
                if existing["source"] in deleted or existing["target"] in deleted:
                    raise ValueError(f"Relationship {edge['name']} refers to a dataset planned for deletion")
            else:
                previous = edges.get(edge_id)
                if previous and (edge["source"], edge["target"]) == (previous["source"], previous["target"]):
                    edge = {**edge, "detail": {**previous["detail"], **edge["detail"]}}
                proposed_joins.append({**edge, "action": action})

    for edge_id, edge in list(edges.items()):
        source, target = edge["source"], edge["target"]
        if (
            source in deleted
            and target not in deleted | replaced_inputs
            and edge["kind"] not in {"join", "reads_table"}
        ):
            raise ValueError(f"{target} still depends on {source}; include its update or deletion in the proposal")
        if source in deleted or target in deleted or (target in replaced_inputs and edge["kind"] != "join"):
            del edges[edge_id]
    for edge in proposal["edges"]:
        if edge["kind"] != "join" and edge["target"] in replaced_inputs:
            if edge["source"] in deleted:
                raise ValueError(f"{edge['target']} refers to an object planned for deletion: {edge['source']}")
            edges[edge["id"]] = edge
    for edge in proposed_joins:
        if edge["source"] in deleted or edge["target"] in deleted:
            raise ValueError(f"Relationship {edge['name']} refers to a dataset planned for deletion")
        detail = edge["detail"]
        if len(detail.get("from_fields", [])) != len(detail.get("to_fields", [])):
            raise ValueError(f"Relationship {edge['name']} requires matching lengths for from_fields and to_fields")
        edges[edge["id"]] = edge
    for node_id in deleted:
        del nodes[node_id]
    table_readers: dict[str, list[str]] = {}
    for edge in edges.values():
        if edge["kind"] == "reads_table":
            table_readers.setdefault(edge["source"], []).append(nodes[edge["target"]]["name"])
    for table_id, readers in table_readers.items():
        table = nodes[table_id]
        nodes[table_id] = {**table, "detail": {**table["detail"], "datasets": readers}}
    return {
        **proposal,
        "nodes": [node for node in nodes.values() if node["kind"] != "physical_table" or node["id"] in table_readers],
        "edges": list(edges.values()),
    }


@dataclass(frozen=True)
class PreparedSemanticPlan:
    path: str
    model_name: str
    model_path: Path | None
    model_revision: str | None
    old_text: str | None
    baseline_text: str
    new_text: str
    revision: int
    counts: dict[str, int]

    def preview(self) -> dict[str, Any]:
        return {"filePath": self.path, "oldText": self.baseline_text, "newText": self.new_text}


def _model_content(path: Path) -> tuple[bytes | None, str | None]:
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None, None
    return content, sha256(content).hexdigest()


def _graph_text(graph: dict) -> str:
    return json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


class SemanticPlanFileTools:
    """Store a reviewable plan graph without approving its design."""

    permission_category = "semantic_tools"
    _MODEL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")

    def __init__(
        self,
        project_root: str | Path,
        *,
        target_state: OsiSemanticModelTargetState | None = None,
        chat_identity: Callable[[], tuple[str | None, str]] | None = None,
        path_allowed: Callable[[Path], bool] | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self._target_state = target_state
        self._chat_identity = chat_identity
        self.path_allowed = path_allowed
        self._plan_ids: dict[str, str] = {}
        self._revisions: dict[str, int] = {}
        self._write_lock = RLock()
        self._review_commits: set[asyncio.Task[FuncToolResult]] = set()
        self.reset_turn()

    def reset_turn(self) -> None:
        with self._write_lock:
            self._baseline: tuple[str | None, dict] | None = None
            self._accepting_reviews = True
            self._plan_id: str | None = None
            self._revision = 0
            self._saved: PreparedSemanticPlan | None = None

    def available_tools(self):
        from datus.tools.func_tool import trans_to_function_tool

        return [trans_to_function_tool(self.write_semantic_model_plan_file)]

    @tool_schema(strict_mode=False)
    @tool_contract(access="write", direct_call=False)
    def write_semantic_model_plan_file(self, model_name: str, plan_json: dict[str, Any] | str) -> FuncToolResult:
        """Convert planning JSON to a graph and write it without changing the model.

        Optional human-review artifact, not a prerequisite for authoring or target
        selection. The plan uses the format below, not OSI dataset/metric objects.
        Saving checks its structure and references; it does not validate the model,
        approve a source choice or execute a query.

        Args:
            model_name: Use the selected name when authoring a model. Standalone plans use this name
                as their identity.
            plan_json: The complete proposal for this turn, not a copy of the existing model.
                Legacy JSON-encoded objects also work.
                Only these top-level keys are accepted (all optional):
                summary (string), datasets, field_decisions, relationships, metrics
                (arrays of objects, default empty). Each dataset, relationship and
                metric requires a unique nonempty name within its array.
                Dataset: name, optional action (create/update/reuse/delete), source_kind
                (physical/query). For physical use source as the table name; for
                query use source_tables as an array of table names and omit source
                entirely (no SELECT in a plan). Create/update with physical/query
                source_kind requires sources.
                Record query rowset_semantics and native_gap or user_requirement.
                Field decision: dataset and field are required; dataset must exist
                in the baseline or proposal. Optional role, group_by and reason are retained.
                Relationship: name, from_dataset, to_dataset are required; both
                datasets must be planned. Optional from_fields/to_fields are arrays
                of field names in matching order; action and reason are retained.
                Metric: name, optional dataset (a planned dataset), kind
                (atomic/parameterized/filter/compose/window), depends_on (array of
                planned metric names, no duplicates), action, role and reason.
                Keep existing objects as reuse unless explicitly changing them. For updates,
                include the resulting dependencies; use delete only when explicitly requested.

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

    def _check_plan_path(self, target: Path) -> None:
        if self.path_allowed is not None and not self.path_allowed(target):
            raise ValueError("Semantic plan is outside scoped_context.project_files")
        if not target.resolve(strict=False).is_relative_to(self.project_root):
            raise ValueError("Semantic plan path must stay inside the project workspace")

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
                metadata = graph.get("plan", {})
                if (
                    graph.get("model") == model_name
                    and metadata.get("id") == path.parent.name
                    and "model_file" not in metadata
                ):
                    candidates.append((path.stat().st_mtime_ns, path.parent.name))
            except (OSError, ValueError, AttributeError):
                continue
        return max(candidates)[1] if candidates else None

    @staticmethod
    def _model_plan_id(model_name: str) -> str:
        return uuid5(NAMESPACE_URL, f"datus:semantic-model-plan:{model_name}").hex

    def _prepare_standalone(self, model_name: str, plan: dict) -> PreparedSemanticPlan:
        """Keep name-based review for hosts that expose the plan tool on its own."""
        plan_id = (
            self._existing_plan_id(model_name) or self._plan_ids.get(model_name) or self._model_plan_id(model_name)
        )
        target = self.project_root / ".datus/semantic-model-plans" / plan_id / "semantic-model-plan.json"
        self._check_plan_path(target)
        old_text = target.read_text(encoding="utf-8") if target.exists() else None
        previous = json.loads(old_text).get("plan", {}).get("revision", 0) if old_text else 0
        revision = max(previous, self._revisions.get(model_name, 0)) + 1
        content = _graph_text(_plan_graph(plan, model_name, plan_id, revision))
        self._plan_ids[model_name] = plan_id
        self._revisions[model_name] = revision
        return PreparedSemanticPlan(
            target.relative_to(self.project_root).as_posix(),
            model_name,
            None,
            None,
            old_text,
            old_text or "",
            content,
            revision,
            {key: len(plan.get(key, [])) for key in ("datasets", "relationships", "metrics")},
        )

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
            try:
                state = self._target_state
                if state is None and self._chat_identity is None:
                    return self._prepare_standalone(model_name, plan)
                if state is None or not state.selected or not state.selected_path or state.last_error_code:
                    raise ValueError("Select the exact OSI semantic-model target before writing a plan")
                if state.selected["semantic_model_name"] != model_name:
                    raise ValueError("The plan must use the selected semantic-model name")
                model_path = Path(state.selected_path).resolve()
                model_file = model_path.relative_to(self.project_root).as_posix()
                if self._chat_identity is None:
                    raise ValueError("A chat session is required for semantic planning")
                owner, chat_id = self._chat_identity()
                if not chat_id:
                    raise ValueError("A chat session is required for semantic planning")
                identity = json.dumps([owner, chat_id, model_file])
                plan_id = uuid5(NAMESPACE_URL, f"datus:chat-semantic-model-plan:{identity}").hex
                if self._plan_id is not None and self._plan_id != plan_id:
                    raise ValueError("The semantic-model target cannot change during plan review")
                target = self.project_root / ".datus/semantic-model-plans" / plan_id / "semantic-model-plan.json"
                self._check_plan_path(target)
                old_text = target.read_text(encoding="utf-8") if target.exists() else None
                previous = json.loads(old_text).get("plan", {}).get("revision", 0) if old_text else 0
                revision = max(previous, self._revision) + 1
                model_content, model_revision = _model_content(model_path)
                if (state.artifact_sha256 or None) != model_revision:
                    raise ValueError(f"{model_file} changed after selection; select the current target again")
                if model_content is None and (state.mode != "planned" or state.selected.get("exists")):
                    raise ValueError(f"Cannot read existing model {model_file}")
                if self._baseline is None:
                    baseline = {
                        "version": 1,
                        "model": model_name,
                        "plan": {"id": plan_id, "model_file": model_file},
                        **_model_graph(
                            model_content,
                            model_name,
                            model_file,
                            repairable=state.mode == "planned" and state.selected.get("repair_required") is True,
                        ),
                    }
                    if previous:
                        baseline["plan"]["revision"] = previous
                else:
                    baseline_revision, baseline = self._baseline
                    if baseline_revision != model_revision:
                        raise ValueError(
                            f"{model_file} changed since this plan's baseline; start a fresh planning turn"
                        )
                graph = _merge_plan_graph(baseline, _plan_graph(plan, model_name, plan_id, revision, baseline))
                graph["plan"]["model_file"] = model_file
                if baseline.get("lineage_incomplete"):
                    notice = "The model needs repair. The baseline shows readable YAML declarations without compiled lineage."
                    graph["plan"]["summary"] = " ".join(filter(None, (graph["plan"]["summary"], notice)))
                content = _graph_text(graph)
            except (
                OSError,
                TypeError,
                ValueError,
                RuntimeError,
                AttributeError,
                KeyError,
                SemanticCoreException,
            ) as exc:
                return FuncToolResult(success=0, error=f"Invalid semantic plan: {exc}")
            self._plan_id = plan_id
            self._revision = revision
            self._baseline = (model_revision, baseline)
            return PreparedSemanticPlan(
                target.relative_to(self.project_root).as_posix(),
                model_name,
                model_path,
                model_revision,
                old_text,
                _graph_text(baseline) if baseline["nodes"] else "",
                content,
                revision,
                {key: len(plan.get(key, [])) for key in ("datasets", "relationships", "metrics")},
            )

    def commit(self, prepared: PreparedSemanticPlan) -> FuncToolResult:
        """Commit exactly the reviewed content, rejecting a stale baseline."""
        target = self.project_root / prepared.path
        with self._write_lock:
            try:
                self._check_plan_path(target)
                with ExitStack() as locks:
                    if prepared.model_path is None:
                        model_lock = target.parent.parent / f"{self._model_plan_id(prepared.model_name)}.model"
                        if not model_lock.resolve(strict=False).is_relative_to(self.project_root):
                            return FuncToolResult(
                                success=0, error="Semantic plan path must stay inside the project workspace"
                            )
                        locks.enter_context(_plan_mutation_lock(model_lock))
                    elif prepared.model_path.parent.is_dir():
                        locks.enter_context(semantic_artifact_lock(prepared.model_path, load_before_lock=False))
                    else:
                        # The project lock also covers the first managed model write without
                        # creating the model directory just to publish a plan.
                        locks.enter_context(
                            ontology_bundle_lock(ontology_dir_for_model(prepared.model_path), create=True)
                        )
                        locks.enter_context(semantic_model_load_lock(prepared.model_path))
                    locks.enter_context(_plan_mutation_lock(target))
                    if prepared.model_path is None:
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
                    if prepared.model_path is not None:
                        _, model_revision = _model_content(prepared.model_path)
                        if model_revision != prepared.model_revision:
                            return FuncToolResult(
                                success=0,
                                error="The semantic model changed during review. Start a fresh planning turn.",
                            )
                    atomic_write_text(target, prepared.new_text)
            except ValueError as exc:
                return FuncToolResult(success=0, error=str(exc))
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Could not write semantic plan: {exc}")
            if prepared.model_path is None:
                previous = self._revisions.get(prepared.model_name, 0)
                self._revisions[prepared.model_name] = max(previous, prepared.revision)
            else:
                self._revision = max(self._revision, prepared.revision)
                self._saved = prepared
            return FuncToolResult(
                result={"path": prepared.path, "revision": prepared.revision, "counts": prepared.counts}
            )

    async def commit_review(self, prepared: PreparedSemanticPlan) -> FuncToolResult:
        """Keep accepted web writes attached to the turn even if the tool task is cancelled."""
        if prepared.model_path is None:
            return await asyncio.to_thread(self.commit, prepared)
        if not self._accepting_reviews:
            return FuncToolResult(success=0, error="The planning turn ended before the plan could be written.")
        task = asyncio.create_task(asyncio.to_thread(self.commit, prepared))
        self._review_commits.add(task)
        task.add_done_callback(self._review_commit_done)
        return await asyncio.shield(task)

    def _review_commit_done(self, task: asyncio.Task[FuncToolResult]) -> None:
        self._review_commits.discard(task)
        if not task.cancelled():
            # The caller can be gone by the time the write fails.
            task.exception()

    async def finish_review_commits(self) -> None:
        """Drain accepted writes before recording the final outcome."""
        self._accepting_reviews = False
        if not self._review_commits:
            return
        pending = asyncio.gather(*self._review_commits, return_exceptions=True)
        cancelled = False
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    def record_outcome(self, *, completed: bool) -> None:
        """Record actual model state after validation/sync, leaving the review snapshot intact."""
        with self._write_lock:
            saved, self._saved = self._saved, None
            if saved is None or saved.model_path is None:
                return
            target = self.project_root / saved.path
            if not target.resolve(strict=False).is_relative_to(self.project_root):
                raise ValueError("Semantic plan path must stay inside the project workspace")
            graph = json.loads(saved.new_text)
            graph["plan"]["execution"] = "completed" if completed else "not_applied"
            if completed:
                content, revision = _model_content(saved.model_path)
                if content is None or self._target_state is None or revision != self._target_state.artifact_sha256:
                    raise ValueError("The validated semantic model changed before its plan outcome could be recorded")
                graph["applied"] = _model_graph(content, graph["model"], graph["plan"]["model_file"])
            with _plan_mutation_lock(target):
                if target.read_text(encoding="utf-8") != saved.new_text:
                    raise ValueError("A newer semantic plan replaced this plan before its outcome could be recorded")
                atomic_write_text(target, _graph_text(graph))
