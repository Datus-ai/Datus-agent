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
    """
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(item, dict) for item in payload):
        return None

    resolved_columns: List[Tuple[str, str]]
    if columns is None:
        resolved_columns = list(_infer_columns(payload))
    else:
        resolved_columns = [(k, label) for k, label in columns]
    if not resolved_columns:
        return None

    table = Table(show_header=True, header_style=header_style, title=title)
    for _, label in resolved_columns:
        table.add_column(str(label))
    for item in payload:
        table.add_row(*(format_cell(item.get(key)) for key, _ in resolved_columns))
    return table


def _infer_columns(rows: Iterable[dict]) -> Iterable[Tuple[str, str]]:
    seen: set = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            yield (key, key)
