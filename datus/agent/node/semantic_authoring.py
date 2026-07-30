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

import asyncio
import hashlib
import json
import re
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

AUTHORING_FORMAT_METRICFLOW = "metricflow"
AUTHORING_FORMAT_OSI = "osi"

_SEMANTIC_AUTHORING_LOCKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Authoring specification skills injected into the system prompt on every run,
# keyed by node name then authoring format. These carry the full YAML format
# spec, so they are host-injected (``REQUIRED_SKILLS`` semantics) rather than
# advertised for LLM-initiated ``load_skill``.
_REQUIRED_AUTHORING_SKILLS: Dict[str, Dict[str, str]] = {
    "gen_semantic_model": {
        AUTHORING_FORMAT_METRICFLOW: "sql-modeling-preflight,metricflow-semantic-authoring",
        AUTHORING_FORMAT_OSI: "sql-modeling-preflight,osi-semantic-authoring",
    },
    "gen_metrics": {
        AUTHORING_FORMAT_METRICFLOW: "sql-modeling-preflight,gen-metrics",
        AUTHORING_FORMAT_OSI: "sql-modeling-preflight,osi-metrics-authoring",
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


def osi_semantic_model_directory(agent_config: Any = None) -> Optional[Path]:
    """Return the active datasource's semantic-model directory."""
    return _osi_semantic_model_dir(agent_config)


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
    """Match a requested table without discarding explicit qualification."""
    left_qualified, left_leaf, left_is_qualified = _table_reference(left)
    right_qualified, right_leaf, right_is_qualified = _table_reference(right)
    if not left_leaf or not right_leaf:
        return False
    if left_is_qualified:
        return right_is_qualified and left_qualified == right_qualified
    return left_leaf == right_leaf


def _model_covers_table(model: Dict[str, Any], table: Any) -> bool:
    references = model.get("table_references") or []
    return any(_table_references_match(table, reference) for reference in references)


def _is_query_backed_dataset(dataset: Dict[str, Any]) -> bool:
    extensions = dataset.get("custom_extensions") or []
    if isinstance(extensions, dict):
        extensions = [extensions]
    for extension in extensions:
        if not isinstance(extension, dict):
            continue
        vendor_name = str(extension.get("vendor_name") or "").strip()
        if vendor_name and vendor_name.upper() != "DATUS":
            continue
        data = extension.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict) and str(data.get("source_type") or "").strip().lower() == "query":
            return True
    return False


def _dataset_table_references(agent_config: Any, dataset: Dict[str, Any], dataset_source: str) -> list[str]:
    if not dataset_source or not _is_query_backed_dataset(dataset):
        return [dataset_source] if dataset_source else []
    references, _source_names, parse_errors = _source_query_table_references(
        agent_config,
        [{"source_sql_name": str(dataset.get("name") or "query_backed_dataset"), "sql": dataset_source}],
    )
    return references if not parse_errors else []


def validate_osi_core_document(document: Any) -> Optional[str]:
    """Validate one parsed document with the OSI package's canonical schema."""
    if not isinstance(document, dict):
        return "YAML document must be an object"
    try:
        from datus_semantic_osi.errors import OSIValidationError
        from datus_semantic_osi.profile import validate_osi_core_schema
    except ImportError as exc:
        return f"OSI schema validator is unavailable: {exc}"

    try:
        validate_osi_core_schema(document)
    except OSIValidationError as exc:
        return str(exc)
    return None


def inspect_osi_semantic_model_inventory(agent_config: Any = None) -> Dict[str, Any]:
    """Inspect every YAML file in the active datasource semantic-model tree.

    ``models`` contains schema-valid targets that metrics may bind. A parsed
    single-model document with a stable name remains in ``recoverable_models``
    when only core-schema validation fails, so semantic authoring can repair it
    without weakening metric binding.
    """
    model_dir = _osi_semantic_model_dir(agent_config)
    if model_dir is None or not model_dir.is_dir():
        return {
            "models": [],
            "recoverable_models": [],
            "issues": [],
            "discovery_warnings": [],
            "files_scanned": 0,
        }

    datasource = str(getattr(agent_config, "current_datasource", "") or "default").strip() or "default"
    model_root = model_dir.resolve(strict=False)
    discovered: list[Dict[str, Any]] = []
    recoverable: list[Dict[str, Any]] = []
    issues: list[Dict[str, Any]] = []
    discovery_warnings: list[Dict[str, Any]] = []
    paths = sorted(
        path
        for path in {*model_dir.rglob("*.yml"), *model_dir.rglob("*.yaml")}
        if "metrics" not in path.relative_to(model_dir).parts
    )
    for path in paths:
        relative_path = path.relative_to(model_dir).as_posix()
        semantic_model_file = f"subject/semantic_models/{datasource}/{relative_path}"
        try:
            absolute_path = path.resolve(strict=True)
            absolute_path.relative_to(model_root)
        except (OSError, ValueError) as exc:
            issues.append(
                {
                    "code": "invalid_semantic_model_path",
                    "semantic_model_file": semantic_model_file,
                    "absolute_path": str(path),
                    "error": str(exc),
                }
            )
            continue
        try:
            content = absolute_path.read_bytes()
            document = yaml.safe_load(content.decode("utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            issues.append(
                {
                    "code": "invalid_yaml",
                    "semantic_model_file": semantic_model_file,
                    "absolute_path": str(path),
                    "error": str(exc),
                }
            )
            continue
        models = document.get("semantic_model") if isinstance(document, dict) else None
        if not isinstance(models, list) or not models:
            issues.append(
                {
                    "code": "invalid_semantic_model_document",
                    "semantic_model_file": semantic_model_file,
                    "absolute_path": str(path),
                    "error": "YAML must contain a non-empty semantic_model list",
                }
            )
            continue
        if len(models) != 1:
            issues.append(
                {
                    "code": "multiple_models_in_file",
                    "semantic_model_file": semantic_model_file,
                    "absolute_path": str(path),
                    "error": "Metric authoring requires exactly one semantic model per YAML file",
                }
            )
            continue
        model = models[0]
        if not isinstance(model, dict):
            issues.append(
                {
                    "code": "invalid_semantic_model_entry",
                    "semantic_model_file": semantic_model_file,
                    "absolute_path": str(path),
                    "model_index": 0,
                    "error": "semantic_model entries must be YAML objects",
                }
            )
            continue
        model_name = str(model.get("name") or "").strip()
        if not model_name:
            issues.append(
                {
                    "code": "missing_semantic_model_name",
                    "semantic_model_file": semantic_model_file,
                    "absolute_path": str(path),
                    "model_index": 0,
                    "error": "semantic_model.name is required",
                }
            )
            continue

        identity = {
            "semantic_model_name": model_name,
            "semantic_model_file": semantic_model_file,
            "absolute_path": str(absolute_path),
            "artifact_sha256": hashlib.sha256(content).hexdigest(),
        }
        schema_error = validate_osi_core_document(document)
        if schema_error:
            issue_code = (
                "osi_schema_validator_unavailable"
                if schema_error.startswith("OSI schema validator is unavailable:")
                else "invalid_osi_core_schema"
            )
            issues.append(
                {
                    "code": issue_code,
                    **identity,
                    "error": schema_error,
                }
            )
            if issue_code == "invalid_osi_core_schema":
                recoverable.append({**identity, "repair_required": True})
            continue

        dataset_summaries: list[Dict[str, str]] = []
        table_references: list[str] = []
        for dataset in model.get("datasets") or []:
            if not isinstance(dataset, dict):
                continue
            dataset_name = str(dataset.get("name") or "").strip()
            dataset_source = str(dataset.get("source") or "").strip()
            query_backed = _is_query_backed_dataset(dataset)
            dataset_summaries.append(
                {
                    key: value
                    for key, value in {
                        "name": dataset_name,
                        "source": dataset_source if not query_backed else "",
                        "description": str(dataset.get("description") or "").strip(),
                    }.items()
                    if value
                }
            )
            dataset_references = _dataset_table_references(agent_config, dataset, dataset_source)
            if dataset_source and query_backed and not dataset_references:
                discovery_warnings.append(
                    {
                        "code": "query_backed_dataset_tables_unknown",
                        "semantic_model_file": semantic_model_file,
                        "absolute_path": str(path),
                        "model_index": 0,
                        "dataset_name": dataset_name,
                        "error": (
                            f"Cannot extract physical tables from query-backed dataset {dataset_name or '<unnamed>'!r}"
                        ),
                    }
                )
            if not dataset_references and dataset_name and not query_backed:
                dataset_references = [dataset_name]
            for table_reference in dataset_references:
                if table_reference not in table_references:
                    table_references.append(table_reference)
        discovered.append(
            {
                **identity,
                "description": str(model.get("description") or "").strip(),
                "datasets": dataset_summaries,
                "table_references": table_references,
            }
        )
    models_by_name: Dict[str, list[Dict[str, Any]]] = {}
    for model in [*discovered, *recoverable]:
        models_by_name.setdefault(model["semantic_model_name"], []).append(model)
    for model_name, duplicates in models_by_name.items():
        if len(duplicates) < 2:
            continue
        for duplicate in duplicates:
            issues.append(
                {
                    "code": "duplicate_semantic_model_name",
                    "semantic_model_name": model_name,
                    "semantic_model_file": duplicate["semantic_model_file"],
                    "absolute_path": duplicate["absolute_path"],
                    "error": (f"semantic_model.name {model_name!r} is declared by multiple YAML files"),
                }
            )
    duplicate_names = {
        issue["semantic_model_name"] for issue in issues if issue.get("code") == "duplicate_semantic_model_name"
    }
    if duplicate_names:
        discovered = [model for model in discovered if model["semantic_model_name"] not in duplicate_names]
        recoverable = [model for model in recoverable if model["semantic_model_name"] not in duplicate_names]
    return {
        "models": discovered,
        "recoverable_models": recoverable,
        "issues": issues,
        "discovery_warnings": discovery_warnings,
        "files_scanned": len(paths),
    }


def discover_osi_semantic_models(agent_config: Any = None) -> list[Dict[str, Any]]:
    """Return filesystem-backed OSI models for compatibility callers."""
    return inspect_osi_semantic_model_inventory(agent_config)["models"]


def _dedupe_table_references(values: Iterable[Any]) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            references.append(text)
    return references


def _source_query_value(source_query: Any, field_name: str) -> Any:
    if isinstance(source_query, dict):
        return source_query.get(field_name)
    return getattr(source_query, field_name, None)


def _source_query_dialect(agent_config: Any) -> str:
    try:
        db_config = agent_config.current_db_config()
    except Exception:
        db_config = None
    candidates = (
        getattr(db_config, "type", ""),
        getattr(db_config, "db_type", ""),
        getattr(db_config, "dialect", ""),
        getattr(agent_config, "db_type", ""),
    )
    for candidate in candidates:
        value = getattr(candidate, "value", candidate)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "snowflake"


def _source_query_parse_dialects(primary_dialect: str) -> list[Optional[str]]:
    candidates: list[Optional[str]] = [primary_dialect]
    if primary_dialect.strip().lower() == "starrocks":
        candidates.append("mysql")
    candidates.append(None)
    return list(dict.fromkeys(candidates))


def _source_query_table_references(
    agent_config: Any,
    source_queries: Iterable[Any],
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Parse authoritative source SQL without consulting its rendered prompt."""
    from datus.utils.sql_utils import extract_table_names_strict

    references: list[str] = []
    source_names: list[str] = []
    parse_errors: list[dict[str, str]] = []
    dialects = _source_query_parse_dialects(_source_query_dialect(agent_config))

    for index, source_query in enumerate(source_queries, 1):
        source_name = str(_source_query_value(source_query, "source_sql_name") or f"sql_{index}").strip()
        sql = str(_source_query_value(source_query, "sql") or "").strip()
        source_names.append(source_name)
        if not sql:
            parse_errors.append({"source_sql_name": source_name, "error": "source SQL is empty"})
            continue
        first_error: Optional[Exception] = None
        for dialect in dialects:
            try:
                references.extend(extract_table_names_strict(sql, dialect=dialect, ignore_empty=True))
                break
            except Exception as exc:
                first_error = first_error or exc
        else:
            parse_errors.append({"source_sql_name": source_name, "error": str(first_error)})

    return _dedupe_table_references(references), source_names, parse_errors


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


def _inventory_path_occupants(inventory: Dict[str, Any], target_file: str) -> list[Dict[str, Any]]:
    """Return every inventory record occupying one semantic-model path."""
    occupants: list[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for records in (
        inventory["models"],
        inventory["recoverable_models"],
        inventory["issues"],
    ):
        for record in records:
            if record.get("semantic_model_file") != target_file:
                continue
            absolute_path = str(record.get("absolute_path") or "")
            if absolute_path in seen_paths:
                continue
            seen_paths.add(absolute_path)
            occupant = dict(record)
            occupant["_identity_known"] = bool(occupant.get("semantic_model_name"))
            occupant.setdefault("semantic_model_name", Path(target_file).stem or "unknown")
            occupants.append(occupant)
    return occupants


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
                "semantic_model_name": model.get("semantic_model_name") or "unknown",
                "semantic_model_file": model["semantic_model_file"],
            }
            for model in models
        ],
        "dimension_tables": dimensions,
    }


def plan_osi_semantic_model_target(
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
    inventory = inspect_osi_semantic_model_inventory(agent_config)
    duplicate_models = [issue for issue in inventory["issues"] if issue.get("code") == "duplicate_semantic_model_name"]
    existing_models = inventory["models"]
    named_models = [*existing_models, *inventory["recoverable_models"], *duplicate_models]
    datasource = str(getattr(agent_config, "current_datasource", "") or "default").strip() or "default"

    explicit_name = _normalize_model_name(semantic_model_name)
    if explicit_name:
        target_file = f"subject/semantic_models/{datasource}/{explicit_name}.yml"
        matches = [
            model for model in named_models if _normalize_model_name(model.get("semantic_model_name")) == explicit_name
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
        occupied_file = _inventory_path_occupants(inventory, target_file)
        if occupied_file:
            reason = (
                "The target filename is already occupied by a differently named semantic model."
                if all(model["_identity_known"] for model in occupied_file)
                else "The target filename already exists but is not a safely recoverable semantic model."
            )
            return _ambiguous_osi_target(
                "explicit_name",
                occupied_file,
                dimensions,
                reason,
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
    same_name = [model for model in named_models if _normalize_model_name(model.get("semantic_model_name")) == new_name]
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
    occupied_file = _inventory_path_occupants(inventory, target_file)
    if occupied_file:
        reason = (
            "The inferred target filename is already occupied by a differently named semantic model."
            if all(model["_identity_known"] for model in occupied_file)
            else "The inferred target filename already exists but is not a safely recoverable semantic model."
        )
        return _ambiguous_osi_target(
            matched_by,
            occupied_file,
            dimensions,
            reason,
        )
    return {
        "semantic_model_name": new_name,
        "semantic_model_file": target_file,
        "exists": False,
        "matched_by": matched_by,
        "dimension_tables": dimensions,
    }


def osi_semantic_model_turn_context(agent_config: Any, user_input: Any) -> str:
    """Render raw request-scoped Ossie authoring intent for the user message.

    Semantic-model intent can change between turns in one persisted session, so
    these values must never be frozen into the session system-prompt snapshot.
    """
    if resolve_authoring_format(agent_config) != AUTHORING_FORMAT_OSI:
        return ""

    requested_name = str(getattr(user_input, "semantic_model_name", "") or "").strip()
    business_domain = str(getattr(user_input, "business_domain", "") or "").strip()
    fact_tables = [
        str(value).strip() for value in (getattr(user_input, "fact_tables", None) or []) if str(value).strip()
    ]
    dimension_tables = [
        str(value).strip() for value in (getattr(user_input, "dimension_tables", None) or []) if str(value).strip()
    ]
    if not (requested_name or business_domain or fact_tables or dimension_tables):
        return ""

    lines = [
        "## Ossie Semantic Model Intent for This Turn",
        "This structured intent is request-scoped and supersedes values from earlier turns.",
    ]
    if requested_name:
        lines.append(f"- Requested semantic model name: `{requested_name}`")
    if business_domain:
        lines.append(f"- Business domain: `{business_domain}`")
    if fact_tables:
        lines.append(f"- Fact tables: `{', '.join(fact_tables)}`")
    if dimension_tables:
        lines.append(f"- Dimension tables: `{', '.join(dimension_tables)}`")
    lines.append("Pass this intent to `plan_osi_semantic_model_target`; only the tool result binds the target.")
    return "\n".join(lines)


def semantic_authoring_lock_key(agent_config: Any = None) -> str:
    """Return the lock key shared by semantic authoring for one datasource."""
    path_manager = getattr(agent_config, "path_manager", None)
    project_root = getattr(path_manager, "project_root", "")
    datasource = str(getattr(agent_config, "current_datasource", "") or "default")
    try:
        project_key = str(Path(project_root).expanduser().resolve(strict=False))
    except TypeError:
        project_key = str(project_root)
    return f"{project_key}:{datasource}"


def _semantic_authoring_lock(agent_config: Any = None) -> asyncio.Lock:
    """Return the event-loop-local lock for one project datasource."""
    loop = asyncio.get_running_loop()
    loop_locks = _SEMANTIC_AUTHORING_LOCKS.setdefault(loop, {})
    return loop_locks.setdefault(semantic_authoring_lock_key(agent_config), asyncio.Lock())


@asynccontextmanager
async def semantic_authoring_guard(agent_config: Any = None):
    """Serialize semantic writes for one project datasource."""
    async with _semantic_authoring_lock(agent_config):
        yield


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
