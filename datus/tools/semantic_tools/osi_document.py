# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Minimal OSI-core document reader for Knowledge Base publication.

Execution validation remains the active semantic engine's responsibility. This
module only turns already-authored OSI YAML into the small object view consumed
by ``GenerationTools`` when it writes semantic and metric records to storage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OsiSource:
    table: str | None = None
    query: str | None = None


@dataclass
class OsiDimension:
    name: str
    expr: str
    type: str = "categorical"
    granularity: str | None = None
    description: str = ""
    ai_context: Any = None


@dataclass
class OsiDataset:
    name: str
    source: OsiSource
    primary_key: list[str] = field(default_factory=list)
    time_dimension: OsiDimension | None = None
    dimensions: list[OsiDimension] = field(default_factory=list)
    fields: list[OsiDimension] = field(default_factory=list)
    description: str = ""
    ai_context: Any = None
    custom_extensions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OsiRelationship:
    name: str
    from_dataset: str
    to_dataset: str
    from_columns: list[str]
    to_columns: list[str]
    type: str = "many_to_one"
    join_type: str | None = None
    ai_context: Any = None


@dataclass
class OsiMetricInput:
    name: str
    alias: str | None = None


@dataclass
class OsiMetric:
    name: str
    description: str = ""
    expression: str | None = None
    dataset: str | None = None
    datasets: list[str] = field(default_factory=list)
    subject_path: list[str] | None = None
    kind: str | None = None
    measures: list[str] = field(default_factory=list)
    inputs: list[str | OsiMetricInput] = field(default_factory=list)
    numerator: str | None = None
    denominator: str | None = None
    window: Any = None
    time_dimension: str | None = None
    unit: str | None = None
    format: str | None = None
    ai_context: Any = None


@dataclass
class OsiDocument:
    name: str
    datasets: list[OsiDataset]
    relationships: list[OsiRelationship]
    metrics: list[OsiMetric]


def _extension_payload(node: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    extensions = node.get("custom_extensions") or []
    if isinstance(extensions, dict):
        extensions = [extensions]
    for extension in extensions:
        if not isinstance(extension, dict):
            continue
        if str(extension.get("vendor_name") or "").strip().upper() != "DATUS":
            continue
        data = extension.get("data")
        if isinstance(data, dict):
            payload.update(data)
        elif isinstance(data, str) and data.strip():
            try:
                decoded = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                payload.update(decoded)
    return payload


def _expression(node: dict[str, Any]) -> str | None:
    expression = node.get("expression")
    if isinstance(expression, str):
        return expression
    if not isinstance(expression, dict):
        return None
    dialects = [item for item in expression.get("dialects") or [] if isinstance(item, dict)]
    for item in dialects:
        if str(item.get("dialect") or "").upper() == "ANSI_SQL" and isinstance(item.get("expression"), str):
            return item["expression"]
    for item in dialects:
        if isinstance(item.get("expression"), str):
            return item["expression"]
    return None


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _source(value: Any) -> OsiSource:
    text = str(value or "").strip()
    if text.lstrip().lower().startswith(("select ", "with ", "(")):
        return OsiSource(query=text)
    return OsiSource(table=text)


def _dimension(field_node: dict[str, Any]) -> OsiDimension:
    payload = _extension_payload(field_node)
    dimension = field_node.get("dimension") or {}
    is_time = isinstance(dimension, dict) and bool(dimension.get("is_time"))
    name = str(field_node.get("name") or "")
    return OsiDimension(
        name=name,
        expr=_expression(field_node) or name,
        type="time" if is_time else str(payload.get("type") or "categorical"),
        granularity=str(payload.get("time_granularity") or payload.get("granularity") or "") or None,
        description=str(field_node.get("description") or ""),
        ai_context=field_node.get("ai_context"),
    )


def _dataset(node: dict[str, Any]) -> OsiDataset:
    payload = _extension_payload(node)
    fields = [item for item in node.get("fields") or [] if isinstance(item, dict) and item.get("name")]
    time_fields = [item for item in fields if bool((item.get("dimension") or {}).get("is_time"))]
    explicit_time = payload.get("time_dimension")
    if isinstance(explicit_time, dict):
        explicit_time = explicit_time.get("name")
    primary_time_name = str(explicit_time or "").rsplit(".", 1)[-1]
    if not primary_time_name and len(time_fields) == 1:
        primary_time_name = str(time_fields[0].get("name") or "")

    primary_time: OsiDimension | None = None
    dimensions: list[OsiDimension] = []
    field_views: list[OsiDimension] = []
    primary_keys = _names(node.get("primary_key"))
    for field_node in fields:
        name = str(field_node.get("name") or "")
        dimension = _dimension(field_node)
        field_views.append(dimension)
        if name == primary_time_name:
            primary_time = dimension
        if name != primary_time_name and name not in primary_keys:
            dimensions.append(dimension)

    return OsiDataset(
        name=str(node.get("name") or ""),
        source=_source(node.get("source")),
        primary_key=primary_keys,
        time_dimension=primary_time,
        dimensions=dimensions,
        fields=field_views,
        description=str(node.get("description") or ""),
        ai_context=node.get("ai_context"),
        custom_extensions=[item for item in node.get("custom_extensions") or [] if isinstance(item, dict)],
    )


def _relationship(node: dict[str, Any]) -> OsiRelationship:
    payload = _extension_payload(node)
    return OsiRelationship(
        name=str(node.get("name") or ""),
        from_dataset=str(node.get("from") or ""),
        to_dataset=str(node.get("to") or ""),
        from_columns=_names(node.get("from_columns")),
        to_columns=_names(node.get("to_columns")),
        type=str(payload.get("type") or "many_to_one"),
        join_type=str(payload.get("join_type") or "") or None,
        ai_context=node.get("ai_context"),
    )


def _metric(node: dict[str, Any], dataset_names: list[str]) -> OsiMetric:
    payload = _extension_payload(node)
    expression = _expression(node)
    datasets = _names(payload.get("datasets"))
    dataset = str(payload.get("dataset") or "") or None
    if dataset:
        datasets = [dataset]
    elif not datasets and expression:
        datasets = [
            name for name in dataset_names if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\.", expression)
        ]
    kind = str(payload.get("metric_kind") or payload.get("metric_type") or "") or None
    if not datasets and len(dataset_names) == 1 and str(kind or "").lower() != "derived":
        datasets = list(dataset_names)
    if dataset is None and len(datasets) == 1:
        dataset = datasets[0]

    inputs: list[str | OsiMetricInput] = []
    for item in payload.get("inputs") or []:
        if isinstance(item, str):
            inputs.append(item)
        elif isinstance(item, dict) and item.get("name"):
            inputs.append(
                OsiMetricInput(
                    name=str(item["name"]),
                    alias=str(item.get("alias") or "") or None,
                )
            )

    subject_path = payload.get("subject_path")
    return OsiMetric(
        name=str(node.get("name") or ""),
        description=str(node.get("description") or ""),
        expression=expression,
        dataset=dataset,
        datasets=datasets,
        subject_path=_names(subject_path) if subject_path is not None else None,
        kind=kind,
        measures=_names(payload.get("measures")),
        inputs=inputs,
        numerator=str(payload.get("numerator") or "") or None,
        denominator=str(payload.get("denominator") or "") or None,
        window=payload.get("window"),
        time_dimension=str(payload.get("time_dimension") or "") or None,
        unit=str(payload.get("unit") or "") or None,
        format=str(payload.get("format") or "") or None,
        ai_context=node.get("ai_context"),
    )


def load_osi_document(path: str, semantic_model_name: str) -> OsiDocument:
    """Read one named OSI core model from one exact publication artifact."""

    model_file = Path(path)
    if not model_file.is_file():
        raise ValueError(f"OSI publication artifact is not a file: {path!r}")

    matches: list[dict[str, Any]] = []
    available: set[str] = set()
    try:
        with model_file.open(encoding="utf-8") as handle:
            documents = list(yaml.safe_load_all(handle))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read OSI model {str(model_file)!r}: {exc}") from exc
    for document in documents:
        if not isinstance(document, dict):
            continue
        models = document.get("semantic_model") or []
        if isinstance(models, dict):
            models = [models]
        for model in models:
            if not isinstance(model, dict):
                continue
            name = str(model.get("name") or "")
            if name:
                available.add(name)
            if name == semantic_model_name:
                matches.append(model)

    if not matches:
        choices = ", ".join(sorted(available)) or "<none>"
        raise ValueError(f"semantic model {semantic_model_name!r} was not found in {path!r}; available: {choices}")
    if len(matches) != 1:
        raise ValueError(
            f"semantic model {semantic_model_name!r} is declared {len(matches)} times in publication artifact {path!r}"
        )

    model = matches[0]
    datasets_raw = [item for item in model.get("datasets") or [] if isinstance(item, dict)]
    relationships_raw = [item for item in model.get("relationships") or [] if isinstance(item, dict)]
    metrics_raw = [item for item in model.get("metrics") or [] if isinstance(item, dict)]
    dataset_names = [str(item.get("name") or "") for item in datasets_raw]
    return OsiDocument(
        name=semantic_model_name,
        datasets=[_dataset(item) for item in datasets_raw],
        relationships=[_relationship(item) for item in relationships_raw],
        metrics=[_metric(item, dataset_names) for item in metrics_raw],
    )
