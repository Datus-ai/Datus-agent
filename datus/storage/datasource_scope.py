# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Row-level datasource scoping helpers for project-scoped storage.

Physical storage namespaces remain project-scoped. Datasource isolation is
represented by columns inside each shared project table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from datus_storage_base.conditions import Node, and_, eq, or_

from datus.utils.exceptions import DatusException, ErrorCode

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

DATASOURCE_ID_COLUMN = "datasource_id"
STORAGE_KEY_COLUMN = "storage_key"
LEGACY_DATASOURCE_ID = ""
LEGACY_STORAGE_KEY_PREFIX = "legacy:"


class DatasourceScope(str):
    """Canonical write identity with an authorized legacy name for reads only."""

    def __new__(cls, value: str, legacy_name: str = "", aliases=()):
        instance = super().__new__(cls, value)
        instance.legacy_name = legacy_name
        instance.legacy_names = tuple(dict.fromkeys([legacy_name, *aliases]))
        return instance


def resolve_datasource_id(agent_config: "AgentConfig", datasource_id: Optional[str] = None) -> str:
    """Return the datasource id required by datasource-scoped KB stores."""

    raw_value = datasource_id if datasource_id is not None else getattr(agent_config, "current_datasource", "")
    resolved = str(raw_value or "").strip()
    if not resolved:
        raise DatusException(
            ErrorCode.STORAGE_INVALID_ARGUMENT,
            message_args={"error_message": "datasource is required for datasource-scoped storage"},
        )
    sources = getattr(getattr(agent_config, "services", None), "datasources", None)
    if isinstance(sources, dict):
        id_matches = (
            [(name, entry) for name, entry in sources.items() if getattr(entry, "datasource_id", None) == resolved]
            if datasource_id is not None
            else []
        )
        if len(id_matches) > 1:
            raise DatusException(ErrorCode.STORAGE_INVALID_ARGUMENT, message="Ambiguous datasource ID")
        candidates = id_matches or [(name, entry) for name, entry in sources.items() if name == resolved]
        for name, entry in candidates:
            stable_id = getattr(entry, "datasource_id", "")
            if isinstance(stable_id, str) and stable_id and resolved in (name, stable_id):
                legacy_names = [name, *getattr(entry, "datasource_aliases", [])]
                unambiguous = [
                    alias
                    for alias in legacy_names
                    if sum(
                        alias == key
                        or alias == getattr(value, "datasource_id", "")
                        or alias in getattr(value, "datasource_aliases", [])
                        for key, value in sources.items()
                    )
                    == 1
                ]
                return DatasourceScope(stable_id, name if name in unambiguous else "", unambiguous)
    return resolved


def datasource_condition(datasource_id: str) -> Node:
    """Build a WHERE condition for the datasource row scope."""

    canonical = eq(DATASOURCE_ID_COLUMN, str(datasource_id))
    aliases = [value for value in getattr(datasource_id, "legacy_names", ()) if value and value != datasource_id]
    if aliases:
        return or_(canonical, *(eq(DATASOURCE_ID_COLUMN, value) for value in aliases))
    return canonical


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
