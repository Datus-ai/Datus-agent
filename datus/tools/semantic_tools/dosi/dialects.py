# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Datus datasource type -> engine dialect name normalization.

Deliberately minimal: only aliases where the two vocabularies differ. An
unknown name returns ``None`` so the engine's own resolution (connection
dialect, else duckdb) decides — a silently wrong dialect would change the
emitted SQL, which is worse than no dialect.
"""

from __future__ import annotations

from typing import Optional

_ALIASES = {
    "postgresql": "postgres",
    "greenplum": "postgres",
}


def normalize_dialect(name: Optional[str]) -> Optional[str]:
    """Normalize a Datus datasource type to an engine dialect name, if known."""
    if not name:
        return None
    lowered = str(name).strip().lower()
    return _ALIASES.get(lowered, lowered) or None


def resolve_engine_dialect(name: Optional[str], engine_dialects: list) -> Optional[str]:
    """The engine dialect for a Datus datasource type, or None if unsupported."""
    normalized = normalize_dialect(name)
    if normalized and normalized in set(engine_dialects):
        return normalized
    return None
