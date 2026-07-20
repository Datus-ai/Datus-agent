# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Semantic authoring format resolution for generation nodes.

Datus can author semantic assets in two formats:

- ``metricflow``: the LLM writes MetricFlow YAML directly. This is the original
  behavior and is left untouched.
- ``osi``: the LLM writes OSI semantic models + Datus business hints, which the
  Datus OSI compiler later lowers to a backend (e.g. MetricFlow). The LLM never
  writes backend YAML.

The format is resolved from the global active semantic adapter so semantic model
generation, metric generation, query, and ask flows stay on one semantic layer
for a project. Legacy node-level semantic format fields are ignored.

Both formats share one system prompt template per node; the format-specific
authoring specification is carried by a *required skill* that the node injects
into the prompt at render time (see ``required_authoring_skills``).
"""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

AUTHORING_FORMAT_METRICFLOW = "metricflow"
AUTHORING_FORMAT_OSI = "osi"

# Authoring specification skills injected into the system prompt on every run,
# keyed by node name then authoring format. These carry the full YAML format
# spec, so they are host-injected (``REQUIRED_SKILLS`` semantics) rather than
# advertised for LLM-initiated ``load_skill``.
_REQUIRED_AUTHORING_SKILLS: Dict[str, Dict[str, str]] = {
    "gen_semantic_model": {
        AUTHORING_FORMAT_METRICFLOW: "metricflow-semantic-authoring",
        AUTHORING_FORMAT_OSI: "osi-semantic-authoring",
    },
    "gen_metrics": {
        AUTHORING_FORMAT_METRICFLOW: "gen-metrics",
        AUTHORING_FORMAT_OSI: "osi-metrics-authoring",
    },
}

# Optional skills advertised in ``<available_skills>`` for LLM-initiated
# loading, keyed the same way. These cover conditional workflows (profiling on
# explicit request, semantic-model repair during metric authoring), so the LLM
# decides per request whether to load them.
_OPTIONAL_AUTHORING_SKILLS: Dict[str, Dict[str, str]] = {
    "gen_semantic_model": {
        AUTHORING_FORMAT_METRICFLOW: "semantic-sql-history-profiler",
        AUTHORING_FORMAT_OSI: "semantic-sql-history-profiler",
    },
    "gen_metrics": {
        AUTHORING_FORMAT_METRICFLOW: "metricflow-semantic-authoring",
        AUTHORING_FORMAT_OSI: "",
    },
}


def _resolve_semantic_adapter(agent_config: Any = None) -> Optional[str]:
    resolver = getattr(agent_config, "resolve_semantic_adapter", None)
    if not callable(resolver):
        return None
    return resolver(None)


def resolve_authoring_format(
    agent_config: Any = None,
    node_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve the semantic authoring format from the global semantic adapter."""
    del node_config

    adapter = _resolve_semantic_adapter(agent_config)

    if adapter and str(adapter).strip().lower() == AUTHORING_FORMAT_OSI:
        return AUTHORING_FORMAT_OSI
    return AUTHORING_FORMAT_METRICFLOW


def resolve_semantic_adapter_type(agent_config: Any = None) -> str:
    """Resolve the active semantic adapter, defaulting to MetricFlow."""
    adapter = _resolve_semantic_adapter(agent_config)
    normalized = str(adapter or "").strip().lower()
    if normalized:
        return normalized
    return AUTHORING_FORMAT_METRICFLOW


def is_osi_authoring(agent_config: Any = None, node_config: Optional[Dict[str, Any]] = None) -> bool:
    """Return ``True`` when this node should author OSI instead of MetricFlow."""
    del node_config
    return resolve_authoring_format(agent_config) == AUTHORING_FORMAT_OSI


def _normalize_model_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def _declared_field_names(value: Any) -> set[str]:
    if is_dataclass(value):
        return {field.name for field in fields(value)}

    for attr_name in ("model_fields", "__fields__"):
        field_map = getattr(value, attr_name, None) or getattr(type(value), attr_name, None)
        if isinstance(field_map, dict):
            return set(field_map)

    annotations = getattr(type(value), "__annotations__", None)
    return set(annotations) if isinstance(annotations, dict) else set()


def _config_field_value(config: Any, field_name: str) -> Any:
    if isinstance(config, dict):
        value = config.get(field_name, "")
    elif field_name in _declared_field_names(config):
        value = getattr(config, field_name, "")
    else:
        return ""
    return "" if callable(value) else value


def default_osi_semantic_model_name(agent_config: Any = None) -> str:
    """Return the default OSI semantic model name for the current authoring scope."""
    candidates = []
    if agent_config is not None:
        runtime_context = {}
        runtime_context_getter = getattr(agent_config, "runtime_db_context", None)
        if callable(runtime_context_getter):
            try:
                runtime_context = runtime_context_getter() or {}
            except Exception:
                runtime_context = {}
        if isinstance(runtime_context, dict):
            candidates.extend(
                [
                    runtime_context.get("database"),
                    runtime_context.get("database_name"),
                    runtime_context.get("schema"),
                    runtime_context.get("db_schema"),
                    runtime_context.get("schema_name"),
                    runtime_context.get("catalog"),
                    runtime_context.get("catalog_name"),
                ]
            )
        try:
            db_config = agent_config.current_db_config()
        except Exception:
            db_config = None
        if db_config is not None:
            candidates.extend(
                [
                    _config_field_value(db_config, "database"),
                    _config_field_value(db_config, "schema"),
                    _config_field_value(db_config, "catalog"),
                ]
            )
        candidates.extend(
            [
                getattr(agent_config, "current_datasource", ""),
                getattr(agent_config, "project_name", ""),
            ]
        )

    for candidate in candidates:
        normalized = _normalize_model_name(candidate)
        if normalized:
            return normalized
    return "semantic_model"


def default_osi_semantic_model_file(agent_config: Any = None) -> str:
    """Return the project-relative default YAML path for OSI domain authoring."""
    datasource = ""
    if agent_config is not None:
        datasource = str(getattr(agent_config, "current_datasource", "") or "").strip()
    if not datasource:
        datasource = "default"
    return f"subject/semantic_models/{datasource}/{default_osi_semantic_model_name(agent_config)}.yml"


def _osi_semantic_model_dir(agent_config: Any = None) -> Optional[Path]:
    """Return the active datasource's semantic-model directory when resolvable."""
    if agent_config is None:
        return None

    datasource = str(getattr(agent_config, "current_datasource", "") or "default").strip() or "default"
    path_manager = getattr(agent_config, "path_manager", None)
    semantic_model_path = getattr(path_manager, "semantic_model_path", None)
    if callable(semantic_model_path):
        try:
            return Path(semantic_model_path(datasource)).expanduser()
        except Exception:
            pass

    project_root = getattr(agent_config, "project_root", None)
    if not isinstance(project_root, (str, Path)):
        project_root = getattr(path_manager, "project_root", None)
    if project_root:
        return Path(str(project_root)).expanduser() / "subject" / "semantic_models" / datasource
    return None


def _table_lookup_names(value: Any) -> set[str]:
    """Return stable lookup keys for a logical or fully-qualified table name."""
    parts = [part.strip().strip('`"[]') for part in re.split(r"[./]", str(value or "").strip())]
    parts = [part for part in parts if part]
    if not parts:
        return set()
    normalized = _normalize_model_name(".".join(parts))
    leaf = _normalize_model_name(parts[-1])
    return {candidate for candidate in (normalized, leaf) if candidate}


def _table_reference(value: Any) -> tuple[str, str, bool]:
    """Return ``(qualified, leaf, is_qualified)`` for safe table matching."""
    parts = [part.strip().strip('`"[]') for part in re.split(r"[./]", str(value or "").strip())]
    parts = [part for part in parts if part]
    if not parts:
        return "", "", False
    normalized_parts = [_normalize_model_name(part) for part in parts]
    leaf = normalized_parts[-1]
    qualified = ".".join(normalized_parts)
    return qualified, leaf, len(parts) > 1


def _table_references_match(left: Any, right: Any) -> bool:
    """Match qualified references exactly, falling back only for unqualified input."""
    left_qualified, left_leaf, left_is_qualified = _table_reference(left)
    right_qualified, right_leaf, right_is_qualified = _table_reference(right)
    if not left_leaf or not right_leaf:
        return False
    if left_is_qualified and right_is_qualified:
        return left_qualified == right_qualified
    return left_leaf == right_leaf


def _model_covers_table(model: Dict[str, Any], table: Any) -> bool:
    references = model.get("table_references") or model.get("dataset_sources") or model.get("dataset_names") or []
    return any(_table_references_match(table, reference) for reference in references)


def discover_osi_semantic_models(agent_config: Any = None) -> list[Dict[str, Any]]:
    """Discover Ossie semantic models already authored for the active datasource.

    The result is intentionally filesystem-backed rather than vector-store-backed:
    target selection must preserve the durable model name and file even before or
    after Knowledge Base synchronization.
    """
    model_dir = _osi_semantic_model_dir(agent_config)
    if model_dir is None or not model_dir.is_dir():
        return []

    datasource = str(getattr(agent_config, "current_datasource", "") or "default").strip() or "default"
    discovered: list[Dict[str, Any]] = []
    paths = sorted([*model_dir.glob("*.yml"), *model_dir.glob("*.yaml")])
    for path in paths:
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        models = document.get("semantic_model") if isinstance(document, dict) else None
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, dict):
                continue
            model_name = str(model.get("name") or "").strip()
            if not model_name:
                continue
            dataset_names: list[str] = []
            dataset_sources: list[str] = []
            table_references: list[str] = []
            table_lookup_names: set[str] = set()
            for dataset in model.get("datasets") or []:
                if not isinstance(dataset, dict):
                    continue
                dataset_name = str(dataset.get("name") or "").strip()
                dataset_source = str(dataset.get("source") or "").strip()
                if dataset_name:
                    dataset_names.append(dataset_name)
                    table_lookup_names.update(_table_lookup_names(dataset_name))
                if dataset_source:
                    dataset_sources.append(dataset_source)
                    table_lookup_names.update(_table_lookup_names(dataset_source))
                table_reference = dataset_source or dataset_name
                if table_reference and table_reference not in table_references:
                    table_references.append(table_reference)
            discovered.append(
                {
                    "semantic_model_name": model_name,
                    "semantic_model_file": f"subject/semantic_models/{datasource}/{path.name}",
                    "absolute_path": str(path),
                    "dataset_names": dataset_names,
                    "dataset_sources": dataset_sources,
                    "table_references": table_references,
                    "table_lookup_names": sorted(table_lookup_names),
                }
            )
    return discovered


def osi_semantic_models_cover_tables(agent_config: Any, tables: Iterable[str]) -> bool:
    """Return whether one existing Ossie model covers every requested table."""
    models = discover_osi_semantic_models(agent_config)
    if not models:
        return False
    requested = [str(table).strip() for table in tables if str(table).strip()]
    return bool(requested) and any(all(_model_covers_table(model, table) for table in requested) for model in models)


def _new_osi_semantic_model_name(business_domain: str, fact_tables: Iterable[str]) -> tuple[str, str]:
    normalized_domain = _normalize_model_name(business_domain)
    if normalized_domain:
        return normalized_domain, "business_domain"

    facts = list(fact_tables)
    if facts:
        table_names = _table_lookup_names(facts[0])
        if not table_names:
            return "", "missing_core_fact_table"
        leaf_name = min(table_names, key=lambda value: (value.count("_"), len(value), value))
        if leaf_name.endswith(("_analytics", "_model", "_semantic_model")):
            return leaf_name, "core_fact_table"
        return f"{leaf_name}_analytics", "core_fact_table"
    return "", "missing_core_fact_table"


def _undiscovered_target_occupant(agent_config: Any, target_file: str) -> Optional[Dict[str, Any]]:
    """Return a fail-closed record when a target path exists but was not discoverable."""
    model_dir = _osi_semantic_model_dir(agent_config)
    if model_dir is None:
        return None
    target_path = model_dir / Path(target_file).name
    if not target_path.exists():
        return None
    return {
        "semantic_model_name": target_path.stem or "unknown",
        "semantic_model_file": target_file,
        "absolute_path": str(target_path),
    }


def _ambiguous_osi_target(
    matched_by: str,
    models: Iterable[Dict[str, Any]],
    dimensions: list[str],
    reason: str,
) -> Dict[str, Any]:
    return {
        "ambiguous": True,
        "matched_by": matched_by,
        "reason": reason,
        "candidates": [
            {
                "semantic_model_name": model["semantic_model_name"],
                "semantic_model_file": model["semantic_model_file"],
            }
            for model in models
        ],
        "dimension_tables": dimensions,
    }


def resolve_osi_semantic_model_target(
    agent_config: Any = None,
    semantic_model_name: str = "",
    business_domain: str = "",
    fact_tables: Optional[Iterable[str]] = None,
    dimension_tables: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Resolve a stable Ossie model name and file from authoring intent.

    An explicit name always wins. Without one, an existing model containing the
    core fact table is reused to keep its durable name. A new model is named by
    business domain when available, then by the first/core fact table. The
    dimension-table list is returned for observability but never participates
    in naming, so adding dimensions cannot rename an existing model.
    """
    facts = [str(value).strip() for value in (fact_tables or []) if str(value).strip()]
    dimensions = [str(value).strip() for value in (dimension_tables or []) if str(value).strip()]
    existing_models = discover_osi_semantic_models(agent_config)
    datasource = str(getattr(agent_config, "current_datasource", "") or "default").strip() or "default"

    explicit_name = _normalize_model_name(semantic_model_name)
    if explicit_name:
        target_file = f"subject/semantic_models/{datasource}/{explicit_name}.yml"
        matches = [
            model
            for model in existing_models
            if _normalize_model_name(model.get("semantic_model_name")) == explicit_name
        ]
        if len(matches) == 1:
            target = dict(matches[0])
            target.update({"exists": True, "matched_by": "explicit_name", "dimension_tables": dimensions})
            return target
        if len(matches) > 1:
            return _ambiguous_osi_target(
                "explicit_name",
                matches,
                dimensions,
                "The explicit semantic model name matches multiple existing files.",
            )
        occupied_file = [model for model in existing_models if model["semantic_model_file"] == target_file]
        if occupied_file:
            return _ambiguous_osi_target(
                "explicit_name",
                occupied_file,
                dimensions,
                "The target filename is already occupied by a differently named semantic model.",
            )
        undiscovered_file = _undiscovered_target_occupant(agent_config, target_file)
        if undiscovered_file:
            return _ambiguous_osi_target(
                "explicit_name",
                [undiscovered_file],
                dimensions,
                "The target filename already exists but does not contain a safely discoverable semantic model.",
            )
        return {
            "semantic_model_name": explicit_name,
            "semantic_model_file": target_file,
            "exists": False,
            "matched_by": "explicit_name",
            "dimension_tables": dimensions,
        }

    fact_matches = [model for model in existing_models if facts and _model_covers_table(model, facts[0])]
    if len(fact_matches) == 1:
        target = dict(fact_matches[0])
        target.update({"exists": True, "matched_by": "existing_fact_table", "dimension_tables": dimensions})
        return target
    if len(fact_matches) > 1:
        return _ambiguous_osi_target(
            "existing_fact_table",
            fact_matches,
            dimensions,
            "The core fact table appears in multiple existing semantic models.",
        )

    new_name, matched_by = _new_osi_semantic_model_name(business_domain, facts)
    if not new_name:
        return _ambiguous_osi_target(
            matched_by,
            [],
            dimensions,
            "A business domain or core fact table is required to name a new semantic model safely.",
        )
    same_name = [
        model for model in existing_models if _normalize_model_name(model.get("semantic_model_name")) == new_name
    ]
    if len(same_name) == 1:
        target = dict(same_name[0])
        target.update({"exists": True, "matched_by": matched_by, "dimension_tables": dimensions})
        return target
    if len(same_name) > 1:
        return _ambiguous_osi_target(
            matched_by,
            same_name,
            dimensions,
            "The inferred semantic model name matches multiple existing files.",
        )
    target_file = f"subject/semantic_models/{datasource}/{new_name}.yml"
    occupied_file = [model for model in existing_models if model["semantic_model_file"] == target_file]
    if occupied_file:
        return _ambiguous_osi_target(
            matched_by,
            occupied_file,
            dimensions,
            "The inferred target filename is already occupied by a differently named semantic model.",
        )
    undiscovered_file = _undiscovered_target_occupant(agent_config, target_file)
    if undiscovered_file:
        return _ambiguous_osi_target(
            matched_by,
            [undiscovered_file],
            dimensions,
            "The inferred target filename already exists but does not contain a safely discoverable semantic model.",
        )
    return {
        "semantic_model_name": new_name,
        "semantic_model_file": target_file,
        "exists": False,
        "matched_by": matched_by,
        "dimension_tables": dimensions,
    }


def required_authoring_skills(agent_config: Any, node_name: str) -> str:
    """Return the host-injected authoring spec skill(s) for a generation node.

    The result is a comma-separated pattern string in the same shape as
    ``AgenticNode.REQUIRED_SKILLS``, derived from the active authoring format.
    """
    authoring_format = resolve_authoring_format(agent_config)
    return _REQUIRED_AUTHORING_SKILLS.get(node_name, {}).get(authoring_format, "")


def default_optional_skills(agent_config: Any, node_name: str) -> str:
    """Return the default ``<available_skills>`` pattern for a generation node.

    These skills stay LLM-loadable because their workflows are conditional; the
    active authoring format decides which variants are visible. Users can still
    override with an explicit ``skills:`` entry in node configuration.
    """
    authoring_format = resolve_authoring_format(agent_config)
    return _OPTIONAL_AUTHORING_SKILLS.get(node_name, {}).get(authoring_format, "")
