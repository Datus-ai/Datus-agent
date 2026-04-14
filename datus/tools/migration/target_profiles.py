"""Target database profiles for cross-database migration.

Encapsulates DDL generation differences between Greenplum and StarRocks,
avoiding scattered if/else branches in agent logic.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Union

from datus.tools.migration.type_mapping import map_columns_between_dialects
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Types considered integer-family for key selection
_INTEGER_TYPES = frozenset({"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "INT2", "INT4", "INT8"})


@dataclass
class GreenplumProfile:
    """Greenplum target profile for DDL generation."""

    schema_name: str = "public"

    def format_table_name(self, table_name: str) -> str:
        """Return schema-qualified table name if not already qualified."""
        if "." in table_name:
            return table_name
        return f"{self.schema_name}.{table_name}"

    def ddl_suffix(self) -> str:
        """Return DDL suffix (empty for v1 — no distribution policy)."""
        return ""


@dataclass
class StarRocksProfile:
    """StarRocks target profile for DDL generation."""

    database: str = ""
    catalog: str = "default_catalog"

    def format_table_name(self, table_name: str) -> str:
        """Return database-qualified table name if not already qualified."""
        if "." in table_name:
            return table_name
        return f"{self.database}.{table_name}"

    def select_key_columns(self, columns: List[dict], max_keys: int = 3) -> List[str]:
        """Select key columns for DUPLICATE KEY using priority rules.

        Priority:
        1. Columns with 'id' or '_id' suffix
        2. INT/BIGINT type columns
        3. Non-nullable columns preferred
        4. Fallback to first column
        """
        if not columns:
            return []

        # Score each column: higher is better
        scored = []
        for col in columns:
            name = col["name"]
            col_type = col.get("type", "").upper()
            base_type = re.sub(r"\(.*\)", "", col_type).strip()
            nullable = col.get("nullable", True)

            score = 0
            # Priority 1: id/_id suffix
            if name.lower() == "id" or name.lower().endswith("_id"):
                score += 100
            # Priority 2: integer type
            if base_type in _INTEGER_TYPES:
                score += 50
            # Priority 3: non-nullable
            if not nullable:
                score += 10

            scored.append((score, name))

        # Sort by score descending, take top max_keys
        scored.sort(key=lambda x: (-x[0], x[1]))

        # If no column scored above 0, fallback to first column
        if scored[0][0] == 0:
            return [columns[0]["name"]]

        return [name for _, name in scored[:max_keys] if _ > 0] or [columns[0]["name"]]

    def ddl_suffix(self, key_columns: List[str]) -> str:
        """Return StarRocks DDL suffix with DUPLICATE KEY and DISTRIBUTED BY HASH."""
        key_str = ", ".join(key_columns)
        return f"\nDUPLICATE KEY({key_str})\nDISTRIBUTED BY HASH({key_str}) BUCKETS 10"


def build_target_ddl(
    source_columns: List[dict],
    source_dialect: str,
    target_dialect: str,
    target_table: str,
    target_profile: Optional[Union[GreenplumProfile, StarRocksProfile]] = None,
) -> str:
    """Build CREATE TABLE DDL for the target database.

    Args:
        source_columns: List of source column defs with name, type, nullable.
        source_dialect: Source database dialect (e.g. "duckdb").
        target_dialect: Target database dialect (e.g. "greenplum", "starrocks").
        target_table: Target table name (may be schema-qualified).
        target_profile: Optional target profile for dialect-specific DDL.

    Returns:
        CREATE TABLE DDL string.

    Raises:
        UnsupportedTypeError: If any column type cannot be mapped.
    """
    # Map types
    mapped_columns = map_columns_between_dialects(source_columns, source_dialect, target_dialect)

    # Determine profile defaults
    if target_profile is None:
        if target_dialect.lower() == "starrocks":
            target_profile = StarRocksProfile()
        else:
            target_profile = GreenplumProfile()

    # Format table name
    qualified_table = target_profile.format_table_name(target_table)

    # Build column definitions
    col_defs = []
    for col in mapped_columns:
        null_clause = "" if col.get("nullable", True) else " NOT NULL"
        col_defs.append(f"  {col['name']} {col['target_type']}{null_clause}")

    columns_sql = ",\n".join(col_defs)

    # Build DDL
    if isinstance(target_profile, StarRocksProfile):
        key_columns = target_profile.select_key_columns(source_columns)
        suffix = target_profile.ddl_suffix(key_columns)
        ddl = f"CREATE TABLE {qualified_table} (\n{columns_sql}\n){suffix}"
    else:
        suffix = target_profile.ddl_suffix()
        ddl = f"CREATE TABLE {qualified_table} (\n{columns_sql}\n){suffix}"

    return ddl
