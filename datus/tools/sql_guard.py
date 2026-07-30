# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""EXPLAIN-based row-count guard for LLM-authored artifact queries.

Estimates a query's cost from the engine's own ``EXPLAIN`` plan (planning only
— the statement is never executed) and lets callers reject statements whose
optimizer estimate blows past a fixed ceiling *before* the raw SQL reaches
``execute_query``.

Motivation: an LLM-authored ``CROSS JOIN`` cartesian product (~9e9 rows)
OOM-killed a StarRocks BE. The result rows were never returned — the *join*
itself exhausted memory — so a post-execution byte cap, a ``LIMIT`` wrapper, or
a ``SELECT COUNT(*)`` pre-check are all useless or actively dangerous (the
COUNT still forces the join to materialize). Only ``EXPLAIN`` inspects the plan
without building it.

The estimate is the **heaviest operator's** cardinality (a scan or join node),
not the final result size — that's deliberate: OOM happens while the heaviest
operator materializes, regardless of any downstream GROUP BY/LIMIT. So the
ceiling is calibrated against per-operator row counts, not result rows.

``MAX_ESTIMATED_ROWS`` sits in the wide gap between the two regimes this guard
separates, for the report/dashboard flows it protects:

* Legitimate report/dashboard queries feed charts. Results are aggregated
  (hundreds to a few thousand rows), and ``save_query`` already caps the stored
  result at 5 MB (~tens of thousands of rows). Their heaviest operator is a base
  table scan — on the mid-size analytical data these flows target, that lands in
  the hundreds-of-thousands-to-low-millions range.
* Pathological queries (an unbounded CROSS JOIN, a missing JOIN key, a
  ``WHERE ... OR 1=1``) multiply into the billions.

50M leaves a 50×+ margin over a legitimate multi-million-row scan while still
catching the ~9e9 incident by ~180×.

Fail-open by design: if ``EXPLAIN`` is unsupported, errors, or its output can't
be parsed, the estimate is ``None`` and the query proceeds. A guard that blocked
on parse failure would reject far more legitimate queries than the rare runaway
it protects against, so parsing is best-effort per dialect and any gap degrades
to "allow".
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

# Ceiling on the optimizer's per-operator row estimate. See the module docstring
# for the calibration rationale (report/dashboard scans vs. cartesian blowups).
MAX_ESTIMATED_ROWS = 50_000_000


# StarRocks / Doris: each plan node prints ``cardinality: N`` (some versions use
# ``cardinality=N``). The largest node cardinality bounds the query's cost — an
# exploding join node carries the ~9e9 estimate we want to catch.
_STARROCKS_CARDINALITY_RE = re.compile(r"cardinality\s*[:=]\s*([0-9]+)", re.IGNORECASE)

# DuckDB: physical plan annotates operators with ``EC: N`` (estimated cardinality).
_DUCKDB_EC_RE = re.compile(r"\bEC:\s*([0-9]+)", re.IGNORECASE)

# PostgreSQL / Redshift: ``(cost=.. rows=N width=..)`` on every plan node. Take
# the max across nodes, not the root's — a GROUP BY/LIMIT shrinks the root's
# ``rows`` while an inner join stays huge, and the join is where OOM happens.
_POSTGRES_ROWS_RE = re.compile(r"\brows=([0-9]+)")


def _flatten_text(explain_rows: List[Any]) -> str:
    """Join an EXPLAIN result set (list of dicts / tuples / scalars) into text."""
    parts: List[str] = []
    for row in explain_rows:
        if isinstance(row, dict):
            parts.extend(str(v) for v in row.values())
        elif isinstance(row, (list, tuple)):
            parts.extend(str(v) for v in row)
        else:
            parts.append(str(row))
    return "\n".join(parts)


def _max_match(pattern: re.Pattern[str], text: str) -> Optional[int]:
    values = [int(m) for m in pattern.findall(text)]
    return max(values) if values else None


def _mysql_rows_product(explain_rows: List[Any]) -> Optional[int]:
    """Multiply the per-row ``rows`` estimates of a tabular MySQL EXPLAIN.

    MySQL's classic EXPLAIN reports rows scanned *per table*; the result-set
    magnitude is roughly their product (each join multiplies), so a two-table
    CROSS JOIN of ~1e5 × ~1e5 surfaces as ~1e10 here — a single ``rows`` column
    would hide it.
    """
    product: Optional[int] = None
    for row in explain_rows:
        if not isinstance(row, dict):
            continue
        raw = next((v for k, v in row.items() if isinstance(k, str) and k.lower() == "rows"), None)
        if raw is None:
            continue
        try:
            n = int(raw)
        except (TypeError, ValueError):
            continue
        product = n if product is None else product * n
    return product


def estimate_rows_from_explain(dialect: str, explain_rows: List[Any]) -> Optional[int]:
    """Parse an EXPLAIN result set into a conservative row-count estimate.

    Returns ``None`` when the dialect is unsupported or the plan yields no
    parseable cardinality — callers treat ``None`` as "allow".
    """
    if not explain_rows:
        return None
    # Adapter names are not always sqlglot/engine-family names. Resolve the
    # adapter-provided parser dialect so PostgreSQL-compatible engines such as
    # Hologres retain the same pre-execution safety guard as PostgreSQL.
    from datus.utils.sql_utils import parse_dialect

    normalized = parse_dialect(dialect or "").lower()
    if normalized in ("starrocks", "doris"):
        return _max_match(_STARROCKS_CARDINALITY_RE, _flatten_text(explain_rows))
    if normalized == "duckdb":
        return _max_match(_DUCKDB_EC_RE, _flatten_text(explain_rows))
    if normalized in ("postgres", "postgresql", "redshift"):
        return _max_match(_POSTGRES_ROWS_RE, _flatten_text(explain_rows))
    if normalized == "mysql":
        return _mysql_rows_product(explain_rows)
    return None


def build_oversize_message(estimated: int, threshold: int) -> str:
    """Actionable rejection text handed back to the authoring LLM.

    Leads with the concrete estimate + ceiling, then rewrites that bound the
    plan's heaviest operator, so the model can fix the SQL and retry rather than
    guess at why the save failed.
    """
    return (
        f"Query rejected before execution: the engine's EXPLAIN plan estimates "
        f"~{estimated:,} rows for its heaviest operator, above the {threshold:,}-row "
        f"safety ceiling. A query this large can exhaust database memory and take the "
        f"backend down (a prior unbounded CROSS JOIN estimated ~9,000,000,000 rows and "
        f"OOM-killed the DB). Rewrite the SQL to bound it before saving:\n"
        f"  1. Verify every JOIN has a real ON/USING key. An unqualified CROSS JOIN, "
        f"or a predicate like `WHERE ... OR 1=1`, disables filtering and multiplies "
        f"row counts.\n"
        f"  2. Aggregate (GROUP BY) or filter (WHERE) down to the rows the report "
        f"actually charts — a visualization needs summarized data, not raw fact rows.\n"
        f"  3. Reach for LIMIT only if it bounds the expensive operator itself (push "
        f"it onto the subquery feeding the join). A LIMIT wrapped around the outer "
        f"query won't clear this check — the heaviest operator materializes before "
        f"the LIMIT applies.\n"
        f"Then call the save tool again."
    )
