# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Raw OSI/DATUS model helpers used by discovery and validation.

The native Dosi engine remains authoritative for validation and execution.
These helpers only read the presentation metadata that is not currently part
of ``Engine.metrics()`` and detect legacy execution hints that Dosi would
otherwise ignore.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

DATUS_VENDOR = "DATUS"
TIME_GRAINS = ("day", "week", "month", "quarter", "year")

# These keys belonged to the retired semantic execution profile. They are
# not native structured-window aliases. Letting Dosi ignore them would silently
# execute the undecorated base aggregate, so every Dosi entry point fails closed.
LEGACY_WINDOW_KEYS = frozenset(
    {
        "grain_to_date",
        "window_aggregation",
        "offset_window",
        "period_over_period",
    }
)


def load_document(path: str | Path) -> dict[str, Any]:
    """Load one OSI YAML/JSON document."""

    model_path = Path(path)
    with model_path.open(encoding="utf-8") as handle:
        if model_path.suffix.lower() == ".json":
            document = json.load(handle)
        else:
            document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"OSI model {str(model_path)!r} must contain one object")
    return document


def iter_semantic_models(document: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for model in document.get("semantic_model") or []:
        if isinstance(model, dict):
            yield model


def iter_named_nodes(
    document: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(carrier, display_name, node)`` for DATUS extension carriers."""

    for model in iter_semantic_models(document):
        model_name = str(model.get("name") or "<unnamed-model>")
        yield "semantic_model", model_name, model
        for dataset in model.get("datasets") or []:
            if not isinstance(dataset, dict):
                continue
            dataset_name = str(dataset.get("name") or "<unnamed-dataset>")
            yield "dataset", dataset_name, dataset
            for field in dataset.get("fields") or []:
                if isinstance(field, dict):
                    field_name = str(field.get("name") or "<unnamed-field>")
                    yield "field", f"{dataset_name}.{field_name}", field
        for relationship in model.get("relationships") or []:
            if isinstance(relationship, dict):
                yield (
                    "relationship",
                    str(relationship.get("name") or "<unnamed-relationship>"),
                    relationship,
                )
        for metric in model.get("metrics") or []:
            if isinstance(metric, dict):
                yield "metric", str(metric.get("name") or "<unnamed-metric>"), metric


def decode_extension_data(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def datus_payload(node: dict[str, Any]) -> dict[str, Any]:
    """Return the first DATUS payload, matching the native engine."""

    extensions = node.get("custom_extensions") or []
    if isinstance(extensions, dict):
        extensions = [extensions]
    for extension in extensions:
        if not isinstance(extension, dict):
            continue
        if str(extension.get("vendor_name") or "").upper() != DATUS_VENDOR:
            continue
        return decode_extension_data(extension.get("data"))
    return {}


def metric_payloads(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for carrier, name, node in iter_named_nodes(document):
        if carrier == "metric" and name and name != "<unnamed-metric>":
            result[name] = datus_payload(node)
    return result


def legacy_window_findings(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return legacy execution hints that are unsafe under native Dosi."""

    findings: list[dict[str, Any]] = []
    for carrier, name, node in iter_named_nodes(document):
        if carrier != "metric":
            continue
        payload = datus_payload(node)
        keys = sorted(LEGACY_WINDOW_KEYS.intersection(payload))
        if isinstance(payload.get("window"), str):
            keys = sorted({*keys, "window"})
        if keys:
            findings.append({"carrier": carrier, "name": name, "keys": keys})
    return findings


def legacy_window_error(findings: list[dict[str, Any]]) -> str:
    details = "; ".join(f"metric {finding['name']!r}: {', '.join(finding['keys'])}" for finding in findings)
    return (
        "legacy Python-OSI window hints are not executable by native Dosi "
        f"({details}). Replace them with native structured window definitions."
    )


def queryable_grains(native_grain: Any) -> list[str]:
    """Return grains at or above a stored/native grain."""

    normalized = str(native_grain or "").strip().lower()
    if not normalized:
        return list(TIME_GRAINS)
    if normalized not in TIME_GRAINS:
        return list(TIME_GRAINS)
    return list(TIME_GRAINS[TIME_GRAINS.index(normalized) :])


def validation_messages(payload: dict[str, Any]) -> list[str]:
    """Flatten blocking validation issues into stable human-readable messages."""

    messages: list[str] = []
    for issue in payload.get("issues") or []:
        if isinstance(issue, dict) and issue.get("severity") == "error":
            messages.append(f"{issue.get('code', 'issue')}: {issue.get('message', '')}")
    for issue in payload.get("compile_errors") or []:
        if isinstance(issue, dict):
            message = f"{issue.get('code', 'compile_error')}: {issue.get('message', '')}"
            if issue.get("hint"):
                message = f"{message} | {issue['hint']}"
            messages.append(message)
    return messages
