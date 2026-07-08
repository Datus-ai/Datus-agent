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

OSI mode uses a *separate* prompt template name (``{node}_osi_system``) so the
default ``{node}_system`` latest-version scan is never affected.
"""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Optional

from datus.utils.loggings import get_logger

AUTHORING_FORMAT_METRICFLOW = "metricflow"
AUTHORING_FORMAT_OSI = "osi"

logger = get_logger(__name__)


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


def osi_template_name(node_name: str) -> str:
    """Return the OSI-mode system prompt template name for a generation node."""
    return f"{node_name}_osi_system"


def osi_prompt_version(agent_config: Any, node_name: str, requested: Optional[str]) -> Optional[str]:
    """Resolve the OSI template version, ignoring versions meant for other templates.

    Callers (e.g. success-story bootstrap) often pin the latest version of the
    *metricflow* template ``{node}_system`` and inject it as ``prompt_version``.
    The OSI template ``{node}_osi_system`` versions independently, so an injected
    metricflow version would not exist here. Honor ``requested`` only when it is a
    real version of the OSI template; otherwise fall back to its latest (``None``).
    """
    if not requested:
        return None
    try:
        from datus.prompts.prompt_manager import get_prompt_manager

        available = get_prompt_manager(agent_config=agent_config).list_template_versions(osi_template_name(node_name))
    except Exception as exc:
        logger.debug(
            "Failed to list OSI prompt template versions; falling back to latest. "
            "node_name=%r template_name=%r agent_config=%r error=%s",
            node_name,
            osi_template_name(node_name),
            agent_config,
            exc,
        )
        available = []
    return requested if requested in available else None
