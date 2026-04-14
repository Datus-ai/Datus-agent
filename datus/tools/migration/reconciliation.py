"""Reconciliation check generation for cross-database migration.

Generates pairs of SQL queries (source vs target) for data reconciliation
after a migration transfer.
"""

import re
from typing import List, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Types considered numeric for aggregate checks
_NUMERIC_TYPE_PATTERN = re.compile(
    r"^(INTEGER|INT|INT[248]?|BIGINT|SMALLINT|TINYINT|FLOAT|FLOAT[48]?|DOUBLE|REAL|DECIMAL|NUMERIC|HUGEINT|LARGEINT)",
    re.IGNORECASE,
)

# Types considered date/time for min/max checks
_DATE_TYPE_PATTERN = re.compile(
    r"^(DATE|TIMESTAMP|DATETIME|TIME)",
    re.IGNORECASE,
)


def _is_numeric_type(col_type: str) -> bool:
    return bool(_NUMERIC_TYPE_PATTERN.match(col_type.strip()))


def _is_date_type(col_type: str) -> bool:
    return bool(_DATE_TYPE_PATTERN.match(col_type.strip()))


def _is_minmax_type(col_type: str) -> bool:
    return _is_numeric_type(col_type) or _is_date_type(col_type)


def build_reconciliation_checks(
    source_table: str,
    target_table: str,
    columns: List[dict],
    key_columns: Optional[List[str]] = None,
) -> List[dict]:
    """Build reconciliation check SQL pairs for source vs target comparison.

    Args:
        source_table: Fully qualified source table name.
        target_table: Fully qualified target table name.
        columns: List of column defs with name, type, nullable.
        key_columns: Optional list of key/PK column names. If None or empty,
                     key-dependent checks (duplicate_key, sample_diff) are skipped.

    Returns:
        List of check dicts, each with: name, source_query, target_query.
    """
    checks = []
    has_keys = bool(key_columns)

    # 1. Row count
    checks.append(
        {
            "name": "row_count",
            "source_query": f"SELECT COUNT(*) AS row_count FROM {source_table}",
            "target_query": f"SELECT COUNT(*) AS row_count FROM {target_table}",
        }
    )

    # 2. Null ratio — for all columns
    nullable_cols = [c for c in columns if c.get("nullable", True)]
    if nullable_cols:
        src_parts = []
        tgt_parts = []
        for col in nullable_cols:
            name = col["name"]
            src_parts.append(f"COUNT(*) - COUNT({name}) AS {name}_null_count")
            tgt_parts.append(f"COUNT(*) - COUNT({name}) AS {name}_null_count")

        src_parts.append("COUNT(*) AS total")
        tgt_parts.append("COUNT(*) AS total")

        checks.append(
            {
                "name": "null_ratio",
                "source_query": f"SELECT {', '.join(src_parts)} FROM {source_table}",
                "target_query": f"SELECT {', '.join(tgt_parts)} FROM {target_table}",
            }
        )

    # 3. Min/max — for numeric and date columns
    minmax_cols = [c for c in columns if _is_minmax_type(c["type"])]
    if minmax_cols:
        src_parts = []
        tgt_parts = []
        for col in minmax_cols:
            name = col["name"]
            src_parts.append(f"MIN({name}) AS {name}_min, MAX({name}) AS {name}_max")
            tgt_parts.append(f"MIN({name}) AS {name}_min, MAX({name}) AS {name}_max")

        checks.append(
            {
                "name": "min_max",
                "source_query": f"SELECT {', '.join(src_parts)} FROM {source_table}",
                "target_query": f"SELECT {', '.join(tgt_parts)} FROM {target_table}",
            }
        )

    # 4. Distinct count — for key columns or all columns
    distinct_cols = key_columns if has_keys else [c["name"] for c in columns]
    if distinct_cols:
        src_parts = [f"COUNT(DISTINCT {c}) AS {c}_distinct" for c in distinct_cols]
        tgt_parts = [f"COUNT(DISTINCT {c}) AS {c}_distinct" for c in distinct_cols]
        checks.append(
            {
                "name": "distinct_count",
                "source_query": f"SELECT {', '.join(src_parts)} FROM {source_table}",
                "target_query": f"SELECT {', '.join(tgt_parts)} FROM {target_table}",
            }
        )

    # 5. Duplicate key — only if key columns provided
    if has_keys:
        key_str = ", ".join(key_columns)
        checks.append(
            {
                "name": "duplicate_key",
                "source_query": (
                    f"SELECT {key_str}, COUNT(*) AS cnt FROM {source_table} "
                    f"GROUP BY {key_str} HAVING COUNT(*) > 1 LIMIT 5"
                ),
                "target_query": (
                    f"SELECT {key_str}, COUNT(*) AS cnt FROM {target_table} "
                    f"GROUP BY {key_str} HAVING COUNT(*) > 1 LIMIT 5"
                ),
            }
        )

    # 6. Sample diff — key-based sample, only if key columns provided
    if has_keys:
        key_order = ", ".join(key_columns)
        all_cols = ", ".join(c["name"] for c in columns)
        checks.append(
            {
                "name": "sample_diff",
                "source_query": (f"SELECT {all_cols} FROM {source_table} ORDER BY {key_order} LIMIT 10"),
                "target_query": (f"SELECT {all_cols} FROM {target_table} ORDER BY {key_order} LIMIT 10"),
            }
        )

    # 7. Numeric aggregate — SUM/AVG for numeric columns
    numeric_cols = [c for c in columns if _is_numeric_type(c["type"])]
    if numeric_cols:
        src_parts = []
        tgt_parts = []
        for col in numeric_cols:
            name = col["name"]
            src_parts.append(f"SUM({name}) AS {name}_sum, AVG({name}) AS {name}_avg")
            tgt_parts.append(f"SUM({name}) AS {name}_sum, AVG({name}) AS {name}_avg")

        checks.append(
            {
                "name": "numeric_aggregate",
                "source_query": f"SELECT {', '.join(src_parts)} FROM {source_table}",
                "target_query": f"SELECT {', '.join(tgt_parts)} FROM {target_table}",
            }
        )

    return checks
