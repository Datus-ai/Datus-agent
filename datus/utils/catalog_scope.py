"""Matching for catalog picker scopes and submitted chat references."""

import re
from collections.abc import Mapping
from typing import Any


def catalog_scope_matches(coordinates: Mapping[str, str], pattern: Mapping[str, str]) -> bool:
    if not pattern.get("table_name"):
        return False

    return all(
        not expected or re.fullmatch(re.escape(expected).replace(r"\*", ".*"), coordinates.get(field, "")) is not None
        for field, expected in pattern.items()
    )


def catalog_scope_tokens(scope: Mapping[str, Any], datasource: str) -> list[str]:
    """An explicit datasource map owns the selection, including empty entries."""
    grouped = scope.get("tables_by_datasource")
    if grouped is None:
        tokens = scope.get("tables")
    else:
        tokens = grouped.get(datasource) if isinstance(grouped, Mapping) else None
    if isinstance(tokens, str):
        tokens = tokens.replace("\n", ",").split(",")
    return [
        entry.strip().lstrip("/")
        for entry in (tokens if isinstance(tokens, list) else [])
        if isinstance(entry, str) and entry.strip().lstrip("/")
    ]
