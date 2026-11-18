# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Ossie document format shared by model editors and lightweight readers."""

from __future__ import annotations

import json
import math
from typing import Any

LEGACY_VERSION = "0.2.0.dev0"
CURRENT_VERSION = "0.3.0.dev0"
LEGACY_EXTENSIONS = "custom_extensions"
EXTENSIONS = "nonportable_extensions"


def extension_entries(node: dict) -> list:
    """Read either envelope without guessing which one wins a collision."""
    if LEGACY_EXTENSIONS in node and EXTENSIONS in node:
        raise ValueError(f"{node.get('name', 'object')}: conflicting {LEGACY_EXTENSIONS} and {EXTENSIONS}")
    return node.get(EXTENSIONS, node.get(LEGACY_EXTENSIONS, []))


def _unique_keys(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate extension object key")
        result[key] = value
    return result


def _check_json(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Extension object keys must be strings")
            _check_json(child)
    elif isinstance(value, list):
        for child in value:
            _check_json(child)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Extension numbers must be finite")
    elif value is not None and not isinstance(value, (str, int)):
        raise ValueError("Extension data must contain only JSON values")


def upgrade_extensions(node: dict, *, legacy_document: bool = False) -> None:
    """Upgrade one owned node in place. Vendor payloads and extra metadata are opaque."""
    entries = extension_entries(node)
    if EXTENSIONS not in node and LEGACY_EXTENSIONS not in node:
        return
    if not isinstance(entries, list):
        raise ValueError("Ossie extensions must be a list of objects")
    upgraded = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Ossie extensions must be a list of objects")
        data = entry.get("data")
        if legacy_document and LEGACY_EXTENSIONS in node and not isinstance(data, str):
            raise ValueError("Legacy extension data must be a JSON object encoded as a string")
        if LEGACY_EXTENSIONS in node and isinstance(data, str):
            try:
                data = json.loads(data, object_pairs_hook=_unique_keys)
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"{entry.get('vendor_name')} extension data is not a JSON object") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{entry.get('vendor_name')} extension data must be an object")
        try:
            _check_json(data)
        except RecursionError as exc:
            raise ValueError("Extension data contains a cycle or is too deeply nested") from exc
        upgraded.append({**entry, "data": data})
    if LEGACY_EXTENSIONS in node:
        del node[LEGACY_EXTENSIONS]
        node[EXTENSIONS] = upgraded


def _children(node: dict, key: str) -> list:
    entries = node.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"{key} must be a list")
    return entries


def upgrade_dataset(dataset: dict) -> None:
    upgrade_extensions(dataset)
    for key in ("fields", "metrics"):
        for node in _children(dataset, key):
            if isinstance(node, dict):
                upgrade_extensions(node)


def upgrade_document(document: Any) -> bool:
    """Normalize an owned, parsed model, mapping or ontology. Return whether its format changed."""
    if not isinstance(document, dict):
        return False
    if not any(key in document for key in ("datasets", "semantic_model", "ontology", "concept_mappings")):
        return False
    version = document.get("version")
    if version not in (LEGACY_VERSION, CURRENT_VERSION):
        raise ValueError(f"Unsupported Ossie version: {version!r}")
    wrapped = "semantic_model" in document
    if wrapped:
        models = document["semantic_model"]
        if (
            set(document) != {"version", "semantic_model"}
            or not isinstance(models, list)
            or len(models) != 1
            or not isinstance(models[0], dict)
            or "version" in models[0]
        ):
            raise ValueError("Ossie wrapper must contain exactly one model")
        del document["semantic_model"]
        document.update(models[0])
    changed = wrapped or version == LEGACY_VERSION
    if "datasets" in document:
        nodes = [document, *_children(document, "metrics"), *_children(document, "relationships")]
        for dataset in _children(document, "datasets"):
            if isinstance(dataset, dict):
                nodes.extend([dataset, *_children(dataset, "fields"), *_children(dataset, "metrics")])
    elif "concept_mappings" in document:
        nodes = [document]
        for key in ("ontology_ref", "semantic_model_ref"):
            ref = document.get(key)
            if isinstance(ref, dict) and isinstance(ref.get("document"), dict):
                changed |= upgrade_document(ref["document"])
    else:
        nodes = []
        for mapping in _children(document, "ontology_mappings"):
            embedded = mapping.get("semantic_model") if isinstance(mapping, dict) else None
            if isinstance(embedded, dict):
                # The deprecated embedded layout permits a model without its own version.
                embedded_version = "version" in embedded
                embedded.setdefault("version", version)
                changed |= upgrade_document(embedded)
                if not embedded_version:
                    del embedded["version"]
    for node in nodes:
        if isinstance(node, dict):
            if version == CURRENT_VERSION and LEGACY_EXTENSIONS in node and not wrapped:
                raise ValueError(f"{node.get('name', 'object')}: {LEGACY_EXTENSIONS} is invalid in Ossie {version}")
            upgrade_extensions(node, legacy_document=True)
    document["version"] = CURRENT_VERSION
    return changed
