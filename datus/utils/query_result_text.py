# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""The text an executed ``query_metrics`` or ``select_records`` returns, read back.

The agent reads the text; code that needs the rows reads the cached result
its ``result_id`` names. ``SemanticTools`` writes it as one ``key: value``
line per fact, a blank line, then the rows as CSV::

    result_id: query_metrics:…
    rows: 71
    warning <code>: <message>; hint: <hint>
    note: <…>
    column <name>: <metric> with <param>=<value>

    <header>
    <first rows>
    ... 51 rows omitted
    <last rows>

Only ``rows`` is always there. A long result shows its first and last rows
around one ``... N rows omitted`` line.
"""

import csv
import io
import re
from dataclasses import dataclass
from typing import Any, List, Optional

_OMITTED_ROWS = re.compile(r"^\.\.\. (\d+) rows? omitted$")
_LEADING_COUNT = re.compile(r"\d+")


def omitted_rows_line(count: int) -> str:
    """The line standing for ``count`` rows left out between the first and last shown."""
    return f"... {count} {'row' if count == 1 else 'rows'} omitted"


def omitted_rows(line: str) -> Optional[int]:
    """How many rows ``line`` stands for, when it is the omitted-rows line."""
    match = _OMITTED_ROWS.match(line.strip())
    return int(match.group(1)) if match else None


@dataclass
class QueryResultText:
    result_id: Optional[str]
    row_count: int
    lines: List[str]
    csv: str

    @property
    def columns(self) -> List[str]:
        header = self.csv.split("\n", 1)[0]
        return next(csv.reader(io.StringIO(header)), []) if header else []

    @property
    def shown_rows(self) -> int:
        """Rows the preview lists, the omitted-rows line not counted."""
        rows = [line for line in self.csv.split("\n")[1:] if line and omitted_rows(line) is None]
        return len(rows)


def parse_query_result_text(text: Any) -> Optional[QueryResultText]:
    """``text`` read as a query result, or None when it is not one."""
    if not isinstance(text, str):
        return None
    head, _, body = text.partition("\n\n")
    lines = head.split("\n")
    fields = {}
    for line in lines:
        key, colon, value = line.partition(": ")
        if colon:
            fields.setdefault(key, value)
    count = _LEADING_COUNT.match(fields.get("rows", ""))
    if count is None:
        return None
    return QueryResultText(
        result_id=fields.get("result_id") or None,
        row_count=int(count.group()),
        lines=lines,
        csv=body,
    )
