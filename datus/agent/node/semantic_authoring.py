# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Semantic authoring format resolution for generation nodes.

Datus can author semantic assets in two formats:

- ``metricflow`` (default): the LLM writes MetricFlow YAML directly. This is the
  original behavior and is left untouched.
- ``osi``: the LLM writes OSI semantic models + Datus business hints, which the
  Datus OSI compiler later lowers to a backend (e.g. MetricFlow). The LLM never
  writes backend YAML.

The format is resolved (in priority order) from:

1. an explicit per-node/workflow ``authoring_format`` in ``node_config``;
2. the active semantic adapter (the consumer of the generated assets) -- when it
   is ``osi``, author OSI;
3. the ``metricflow`` default.

OSI mode uses a *separate* prompt template name (``{node}_osi_system``) so the
default ``{node}_system`` latest-version scan is never affected.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from datus.utils.loggings import get_logger

AUTHORING_FORMAT_METRICFLOW = "metricflow"
AUTHORING_FORMAT_OSI = "osi"

logger = get_logger(__name__)


def resolve_authoring_format(
    agent_config: Any = None,
    node_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve the semantic authoring format. Defaults to ``metricflow``."""
    if node_config:
        explicit = node_config.get("authoring_format")
        if explicit:
            normalized = str(explicit).strip().lower()
            if normalized in (AUTHORING_FORMAT_METRICFLOW, AUTHORING_FORMAT_OSI):
                return normalized

    adapter: Optional[str] = None
    adapter_type: Optional[str] = None
    if node_config:
        adapter_type = node_config.get("semantic_adapter") or node_config.get("adapter_type")
    if agent_config is not None and hasattr(agent_config, "resolve_semantic_adapter"):
        try:
            adapter = agent_config.resolve_semantic_adapter(adapter_type)
        except Exception as exc:
            logger.debug(
                "Failed to resolve semantic adapter for authoring format; "
                "falling back to metricflow. adapter_type=%r agent_config=%r error=%s",
                adapter_type,
                agent_config,
                exc,
            )
            adapter = None

    if adapter and str(adapter).strip().lower() == AUTHORING_FORMAT_OSI:
        return AUTHORING_FORMAT_OSI
    return AUTHORING_FORMAT_METRICFLOW


def is_osi_authoring(agent_config: Any = None, node_config: Optional[Dict[str, Any]] = None) -> bool:
    """Return ``True`` when this node should author OSI instead of MetricFlow."""
    return resolve_authoring_format(agent_config, node_config) == AUTHORING_FORMAT_OSI


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
