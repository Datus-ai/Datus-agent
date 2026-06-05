# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers for deriving backend storage namespaces.

The storage registry still names its routing argument ``project`` for backward
compatibility, but datasource-bound KB stores need a narrower backend namespace
than the workspace project.  These helpers keep that derivation centralized.

Datasource-scoped KB stores use ``datasource_storage_namespace()`` so storage,
subject tree rows, and vector rows do not leak across datasources in the same
workspace project. Project-scoped stores use the project namespace directly or
compose their own explicit sub-namespace.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Optional

from datus.utils.exceptions import DatusException, ErrorCode

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

# Informational classification used by tests and maintainers. Runtime routing
# still goes through the helper functions below.
DATASOURCE_SCOPED_KB_STORES = frozenset(
    {
        "ext_knowledge",
        "metric",
        "reference_sql",
        "reference_template",
        "schema_metadata",
        "semantic_model",
        "subject_tree",
    }
)
PROJECT_SCOPED_STORES = frozenset(
    {
        "document",
        "feedback",
        "knowledge_provenance",
        "task",
    }
)

_SAFE_NAMESPACE_RE = re.compile(r"[^A-Za-z0-9_]")
_MAX_NAMESPACE_LEN = 200
_HASH_LEN = 8


def _hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:_HASH_LEN]


def safe_storage_namespace_token(value: object, fallback: str = "default") -> str:
    """Return a backend-safe namespace token.

    The token is intentionally stricter than filesystem path segments so it also
    works for PostgreSQL schema names in physical isolation mode.
    """

    raw = str(value or "").strip()
    normalized = _SAFE_NAMESPACE_RE.sub("_", raw)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        normalized = fallback
    if not (normalized[0].isalpha() or normalized[0] == "_"):
        normalized = f"n_{normalized}"

    needs_digest = normalized != raw or len(normalized) > _MAX_NAMESPACE_LEN
    digest = _hash(raw or fallback)
    if len(normalized) > _MAX_NAMESPACE_LEN:
        keep = _MAX_NAMESPACE_LEN - _HASH_LEN - 1
        normalized = f"{normalized[:keep]}_{digest}"
    elif needs_digest:
        suffix = f"_{digest}"
        if len(normalized) + len(suffix) > _MAX_NAMESPACE_LEN:
            keep = _MAX_NAMESPACE_LEN - len(suffix)
            normalized = normalized[:keep]
        normalized = f"{normalized}{suffix}"
    return normalized


def project_storage_namespace(agent_config: "AgentConfig") -> str:
    """Return the backend namespace for project-scoped storage."""

    return safe_storage_namespace_token(getattr(agent_config, "project_name", ""), fallback="project")


def resolve_datasource_id(agent_config: "AgentConfig", datasource: Optional[str] = None) -> str:
    """Return the datasource id required by datasource-scoped KB storage."""

    datasource_name = datasource if datasource is not None else getattr(agent_config, "current_datasource", "")
    datasource_id = str(datasource_name or "").strip()
    if not datasource_id:
        raise DatusException(
            ErrorCode.STORAGE_INVALID_ARGUMENT,
            message_args={"error_message": "datasource is required for datasource-scoped storage"},
        )
    return datasource_id


def resolve_datasource_scope(agent_config: "AgentConfig", datasource: Optional[str] = None) -> tuple[str, str]:
    """Return ``(datasource_id, storage_namespace)`` for datasource-scoped KB storage."""

    datasource_id = resolve_datasource_id(agent_config, datasource)
    return datasource_id, datasource_storage_namespace(agent_config, datasource_id)


def datasource_storage_namespace(agent_config: "AgentConfig", datasource: Optional[str] = None) -> str:
    """Return the backend namespace for datasource-scoped KB storage."""

    project = project_storage_namespace(agent_config)
    datasource_name = resolve_datasource_id(agent_config, datasource)
    ds_token = safe_storage_namespace_token(datasource_name, fallback="datasource")
    namespace = f"{project}__ds__{ds_token}"
    if len(namespace) <= _MAX_NAMESPACE_LEN:
        return namespace
    digest = _hash(namespace)
    keep = _MAX_NAMESPACE_LEN - _HASH_LEN - 1
    return f"{namespace[:keep]}_{digest}"
