# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Build combined RAG scope filters for sub-agent + request-context isolation.

Extracted from ``StorageCacheHolder._build_scope_filter`` so that RAG
classes can construct their own scoped views without depending on the
(now-removed) ``StorageCache`` layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from datus_storage_base.conditions import Node, and_

from datus.schemas.agent_models import SubAgentConfig
from datus.storage.scoped_filter import ScopedFilterBuilder
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig
    from datus.storage.base import BaseEmbeddingStore

logger = get_logger(__name__)


def build_rag_scope(
    agent_config: "AgentConfig",
    sub_agent_name: Optional[str],
    storage: "BaseEmbeddingStore",
    check_scope_attr: str,
) -> Optional[Node]:
    """Build a combined RAG scope filter.

    Combines:
    1. Sub-agent scoped context filter (table filter or subject filter)
    2. Request-context scope filter (fields listed in ``scope_fields``,
       e.g. workspace_id)

    Args:
        agent_config: Current agent configuration.
        sub_agent_name: Sub-agent name, or None for global access.
        storage: The storage instance (needed for subject_tree access).
        check_scope_attr: Attribute to check in scoped_context
                          (e.g. "tables", "metrics", "sqls", "ext_knowledge").

    Returns:
        Combined filter Node, or None if no scoping is needed.
    """
    from datus.storage.registry import build_scope_filter_from_context

    sub_agent_filter = _build_sub_agent_filter(agent_config, sub_agent_name, storage, check_scope_attr)
    context_filter = build_scope_filter_from_context(agent_config)

    if sub_agent_filter is None and context_filter is None:
        return None
    if sub_agent_filter is not None and context_filter is not None:
        return and_(sub_agent_filter, context_filter)
    return sub_agent_filter or context_filter


def _build_sub_agent_filter(
    agent_config: "AgentConfig",
    sub_agent_name: Optional[str],
    storage: "BaseEmbeddingStore",
    check_scope_attr: str,
) -> Optional[Node]:
    """Build scope filter from sub-agent's scoped context."""
    if not sub_agent_name:
        return None

    raw_config = agent_config.sub_agent_config(sub_agent_name)
    if not raw_config:
        return None

    sub_agent_config = SubAgentConfig.model_validate(raw_config)
    if not sub_agent_config.has_scoped_context_by(check_scope_attr):
        return None

    scope_value = getattr(sub_agent_config.scoped_context, check_scope_attr, None)
    if not scope_value:
        return None

    if check_scope_attr == "tables":
        dialect = getattr(agent_config, "db_type", "")
        return ScopedFilterBuilder.build_table_filter(scope_value, dialect)
    elif check_scope_attr in ("metrics", "sqls", "ext_knowledge"):
        subject_tree = getattr(storage, "subject_tree", None)
        if subject_tree is None:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_VALIDATION_FAILED,
                message=(
                    f"Cannot build scope filter for sub-agent '{sub_agent_name}' "
                    f"(scope attr='{check_scope_attr}'): storage has no subject_tree. "
                    "Refusing to return unscoped storage."
                ),
            )
        return ScopedFilterBuilder.build_subject_filter(scope_value, subject_tree)

    return None
