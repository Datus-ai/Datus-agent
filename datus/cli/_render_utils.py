# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared Rich rendering helpers for CLI commands.

Row-shaped data (list of dicts, whether from ``.tables`` / ``.databases``
or ``.<service>.list_*``) should share one table-rendering implementation.
This module holds that helper so individual command modules don't each
inline their own ``Table()`` construction with drifting styles.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from rich.table import Table


def format_cell(value: Any) -> str:
    """Convert a cell value to the string shown in a Rich Table cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        # Nested structures render as compact inline JSON — the table is a
        # scanning aid, not a place to unfold trees.
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def build_row_table(
    payload: Any,
    *,
    title: Optional[str] = None,
    columns: Optional[Sequence[Tuple[str, str]]] = None,
    header_style: str = "bold green",
    hide_empty_columns: Optional[bool] = None,
) -> Optional[Table]:
    """Build a Rich ``Table`` from a list-of-dict payload.

    Returns ``None`` when the payload doesn't match — empty list, non-list,
    or items that aren't dicts. Callers fall back to their own rendering
    (JSON, "Empty set.", etc.) in that case.

    ``columns`` is an ordered sequence of ``(key, display_label)`` tuples.
    When omitted, the column set is inferred from the union of dict keys
    in the order they first appear — handy for arbitrary rows returned
    by service adapters where the schema isn't known statically. When
    provided, callers pick which keys to expose and how to label them,
    which preserves existing command UX like
    ``"Logic Name(Used for switch)"`` from ``.databases``.

    ``hide_empty_columns`` drops columns whose value is "empty" for every
    row (``None`` / ``""`` / ``[]`` / ``{}``). The default is tied to
    whether columns are inferred: list-of-dict service output routinely
    carries placeholder fields that are only populated by the *detail*
    endpoint (``DashboardInfo.chart_ids`` in Superset / Grafana
    ``list_dashboards`` → always ``[]``), so pruning noise columns is the
    right default for the inference path. Explicit columns are left
    alone — the caller hand-picked them, don't second-guess.
    """
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(item, dict) for item in payload):
        return None

    resolved_columns: List[Tuple[str, str]]
    if columns is None:
        resolved_columns = list(_infer_columns(payload))
        if hide_empty_columns is None:
            hide_empty_columns = True
    else:
        resolved_columns = [(k, label) for k, label in columns]
        if hide_empty_columns is None:
            hide_empty_columns = False
    if not resolved_columns:
        return None

    if hide_empty_columns:
        non_empty_keys = _collect_non_empty_keys(payload)
        resolved_columns = [(key, label) for key, label in resolved_columns if key in non_empty_keys]
        if not resolved_columns:
            return None

    table = Table(show_header=True, header_style=header_style, title=title)
    for _, label in resolved_columns:
        table.add_column(str(label))
    for item in payload:
        table.add_row(*(format_cell(item.get(key)) for key, _ in resolved_columns))
    return table


def _is_empty_value(value: Any) -> bool:
    """True for values that add no information to a table cell."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _collect_non_empty_keys(rows: Iterable[dict]) -> set:
    """Return the set of dict keys that have a non-empty value in any row."""
    keys: set = set()
    for row in rows:
        for key, value in row.items():
            if key in keys:
                continue
            if not _is_empty_value(value):
                keys.add(key)
    return keys


def _infer_columns(rows: Iterable[dict]) -> Iterable[Tuple[str, str]]:
    seen: set = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            yield (key, key)
