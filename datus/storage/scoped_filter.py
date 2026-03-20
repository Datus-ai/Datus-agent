# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Build LanceDB WHERE filters from ScopedContext for sub-agent storage scoping.

Instead of copying data into separate sub-agent directories, sub-agents
apply a scope filter at query time against the shared global storage.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from datus_storage_base.conditions import Node, and_, eq, like, or_

from datus.tools.db_tools import connector_registry
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.reference_paths import split_reference_path

logger = get_logger(__name__)


class ScopedFilterBuilder:
    """Build LanceDB WHERE filters from ScopedContext attributes."""

    @staticmethod
    def build_table_filter(tables_str: str, dialect: str = "") -> Optional[Node]:
        """Build a filter for metadata/semantic_model stores from table identifiers.

        Args:
            tables_str: Comma/newline-separated table identifiers (e.g. "public.users, orders")
            dialect: Database dialect (e.g. DBType.SQLITE) for field-order resolution

        Returns:
            Combined OR condition, or None if no valid tokens
        """
        tokens = _split_csv(tables_str)
        if not tokens:
            return None

        conditions: List[Node] = []
        for token in tokens:
            cond = _table_condition_for_token(token, dialect)
            if cond is not None:
                conditions.append(cond)

        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else or_(*conditions)

    @staticmethod
    def build_subject_filter(paths_str: str, subject_tree) -> Optional[Node]:
        """Build a filter for metrics/reference_sql/ext_knowledge from subject paths.

        Mirrors the two-pass logic in
        ``BaseSubjectEmbeddingStore.search_with_subject_filter``:

        1. Try all parts as a full subject path (``descendant=True``).
        2. If that yields no matches **and** ``len(parts) > 1``, fall back to
           treating ``parts[:-1]`` as the path and ``parts[-1]`` as a name
           filter (``descendant=False``, ``name LIKE '…'``).

        Multiple tokens are OR'd together.

        Args:
            paths_str: Comma/newline-separated subject paths
                       (e.g. ``"Finance.Revenue, Sales.total_revenue"``)
            subject_tree: SubjectTreeStore instance used to resolve paths to
                          node IDs

        Returns:
            Combined condition, or None if nothing resolved
        """
        tokens = _split_csv(paths_str)
        if not tokens:
            return None

        token_conditions: List[Node] = []
        for token in tokens:
            # Normalize '/' separators to '.' before parsing so paths like
            # "Finance/Revenue" are handled consistently with "Finance.Revenue".
            normalized_token = token.replace("/", ".")
            parts = split_reference_path(normalized_token)
            if not parts:
                continue

            cond = _subject_condition_for_parts(parts, subject_tree)
            if cond is not None:
                token_conditions.append(cond)

        if not token_conditions:
            return None
        return token_conditions[0] if len(token_conditions) == 1 else or_(*token_conditions)


def _split_csv(value: Optional[str]) -> List[str]:
    """Split a comma/newline-separated string into trimmed, non-empty tokens."""
    if not value:
        return []
    tokens = [t.strip() for t in str(value).replace("\n", ",").split(",")]
    seen = set()
    result: List[str] = []
    for t in tokens:
        if t and t not in seen:
            result.append(t)
            seen.add(t)
    return result


def _build_id_condition(ids: List[int]) -> Node:
    """Build a subject_node_id equality/OR condition from a list of IDs."""
    unique_ids = list(dict.fromkeys(ids))
    if len(unique_ids) == 1:
        return eq("subject_node_id", unique_ids[0])
    return or_(*[eq("subject_node_id", sid) for sid in unique_ids])


def _subject_condition_for_parts(parts: List[str], subject_tree) -> Optional[Node]:
    """Build a filter for a single subject token using two-pass resolution.

    Pass 1: treat all *parts* as a full subject path (descendant=True).
    Pass 2 (fallback): if pass 1 yields nothing and len(parts) > 1,
            treat parts[:-1] as the path and parts[-1] as the item name.
    """
    # Pass 1 - full path
    matched_ids = subject_tree.get_matched_children_id(parts, True)
    if matched_ids:
        return _build_id_condition(matched_ids)

    # Pass 2 - last component might be a name
    if len(parts) > 1:
        path, name = parts[:-1], parts[-1]
        matched_ids = subject_tree.get_matched_children_id(path, False)
        if matched_ids:
            return and_(_build_id_condition(matched_ids), _value_condition("name", name))

    return None


def _value_condition(field: str, value: str) -> Node:
    """Build an eq or LIKE condition depending on wildcard presence."""
    value = value.strip()
    if not value:
        return eq(field, "")
    if "*" in value:
        return like(field, value)
    return eq(field, value)


def _table_condition_for_token(token: str, dialect: str = "") -> Optional[Node]:
    """Parse a single table identifier token into a LanceDB condition.

    Reuses the same right-aligned field mapping logic from SubAgentBootstrapper.
    """
    parts = [p.strip() for p in token.split(".") if p.strip()]
    if not parts:
        return None

    field_order: List[str] = []
    if connector_registry.support_catalog(dialect):
        field_order.append("catalog_name")
    if connector_registry.support_database(dialect) or dialect == DBType.SQLITE:
        field_order.append("database_name")
    if connector_registry.support_schema(dialect):
        field_order.append("schema_name")
    field_order.append("table_name")

    values: Dict[str, str] = {f: "" for f in field_order}
    num_fields = len(field_order)
    trimmed_parts = parts[-num_fields:]
    start_field_idx = max(0, num_fields - len(trimmed_parts))
    for i, part in enumerate(trimmed_parts):
        field_idx = start_field_idx + i
        if field_idx < num_fields:
            values[field_order[field_idx]] = part

    conditions: List[Node] = []
    for field, value in values.items():
        if not value:
            continue
        conditions.append(_value_condition(field, value))

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else and_(*conditions)
