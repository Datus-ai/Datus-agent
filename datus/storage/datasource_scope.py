# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Row-level datasource scoping helpers for project-scoped storage.

Physical storage namespaces remain project-scoped. Datasource isolation is
represented by columns inside each shared project table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from datus_storage_base.conditions import Node, and_, eq

from datus.utils.exceptions import DatusException, ErrorCode

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

DATASOURCE_ID_COLUMN = "datasource_id"
STORAGE_KEY_COLUMN = "storage_key"
LEGACY_DATASOURCE_ID = ""
LEGACY_STORAGE_KEY_PREFIX = "legacy:"


def resolve_datasource_id(agent_config: "AgentConfig", datasource_id: Optional[str] = None) -> str:
    """Return the datasource id required by datasource-scoped KB stores."""

    raw_value = datasource_id if datasource_id is not None else getattr(agent_config, "current_datasource", "")
    resolved = str(raw_value or "").strip()
    if not resolved:
        raise DatusException(
            ErrorCode.STORAGE_INVALID_ARGUMENT,
            message_args={"error_message": "datasource is required for datasource-scoped storage"},
        )
    return resolved


def datasource_condition(datasource_id: str) -> Node:
    """Build a WHERE condition for the datasource row scope."""

    return eq(DATASOURCE_ID_COLUMN, datasource_id)


def combine_conditions(conditions: Iterable[Optional[Node]]) -> Optional[Node]:
    """Combine non-empty conditions with AND."""

    active = [condition for condition in conditions if condition is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return and_(*active)


def build_storage_key(datasource_id: str, business_id: Any) -> str:
    """Build an internal datasource-scoped unique key for vector upserts."""

    row_id = str(business_id or "").strip()
    if not row_id:
        raise DatusException(
            ErrorCode.STORAGE_INVALID_ARGUMENT,
            message_args={"error_message": "business id is required to build storage_key"},
        )
    datasource = str(datasource_id or "").strip()
    if datasource:
        return f"{datasource}:{row_id}"
    return f"{LEGACY_STORAGE_KEY_PREFIX}{row_id}"


def add_datasource_scope_to_rows(
    rows: List[Dict[str, Any]],
    datasource_id: str,
    *,
    id_field: str = "id",
) -> List[Dict[str, Any]]:
    """Return copies of rows with datasource_id and storage_key populated."""

    scoped_rows: List[Dict[str, Any]] = []
    for row in rows:
        scoped = dict(row)
        scoped[DATASOURCE_ID_COLUMN] = datasource_id
        if id_field and scoped.get(id_field) not in (None, ""):
            scoped[STORAGE_KEY_COLUMN] = build_storage_key(datasource_id, scoped[id_field])
        scoped_rows.append(scoped)
    return scoped_rows
