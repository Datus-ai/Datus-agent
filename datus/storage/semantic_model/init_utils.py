# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers for ``init_*_semantic_model_async`` build-mode handling.

Mirrors the shape of :mod:`datus.storage.reference_sql.init_utils` —
``exists_semantic_models`` returns a set of identifiers already present
in the store; the empty set means the caller can proceed with the
``overwrite`` flow.
"""

from __future__ import annotations

from typing import Set

from datus.storage.semantic_model.store import SemanticModelRAG


def exists_semantic_models(rag: SemanticModelRAG, build_mode: str = "overwrite") -> Set[str]:
    """Return existing table-level semantic model identifiers.

    For ``build_mode="overwrite"`` the result is always empty so the
    caller treats the store as if no models exist (full regeneration).
    For ``build_mode="incremental"`` it returns the set of ``table_name``
    values already indexed; if non-empty, the caller is expected to skip
    the LLM call entirely.
    """
    existing: Set[str] = set()
    if build_mode == "overwrite":
        return existing
    if build_mode == "incremental":
        for row in rag.search_all(""):
            name = row.get("table_name") or row.get("name")
            if name:
                existing.add(str(name))
    return existing
