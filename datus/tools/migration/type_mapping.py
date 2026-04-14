"""Cross-dialect column type mapping for database migration.

Maps DuckDB column types to target dialects (Greenplum, StarRocks).
Raises UnsupportedTypeError for types that cannot be mapped.
"""

import re
from typing import List, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Regex to extract base type and optional parameters, e.g. "DECIMAL(10,2)" -> ("DECIMAL", "(10,2)")
_TYPE_PARAM_RE = re.compile(r"^(\w[\w\s]*)(\(.*\))?$")

# Types that are not mappable in v1
_UNSUPPORTED_TYPES = frozenset(
    {
        "LIST",
        "STRUCT",
        "MAP",
        "UNION",
        "BLOB",
        "BYTEA",
        "GEOMETRY",
        "POINT",
        "LINESTRING",
        "POLYGON",
    }
)

# DuckDB -> Greenplum base type mapping
_DUCKDB_TO_GREENPLUM = {
    "VARCHAR": "VARCHAR",
    "TEXT": "TEXT",
    "STRING": "TEXT",
    "CHAR": "CHAR",
    "INTEGER": "INTEGER",
    "INT": "INTEGER",
    "INT4": "INTEGER",
    "BIGINT": "BIGINT",
    "INT8": "BIGINT",
    "SMALLINT": "SMALLINT",
    "INT2": "SMALLINT",
    "TINYINT": "SMALLINT",
    "DOUBLE": "DOUBLE PRECISION",
    "FLOAT8": "DOUBLE PRECISION",
    "FLOAT": "REAL",
    "FLOAT4": "REAL",
    "REAL": "REAL",
    "DECIMAL": "NUMERIC",
    "NUMERIC": "NUMERIC",
    "BOOLEAN": "BOOLEAN",
    "BOOL": "BOOLEAN",
    "DATE": "DATE",
    "TIMESTAMP": "TIMESTAMP",
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMPTZ": "TIMESTAMP WITH TIME ZONE",
    "TIME": "TIME",
    "UUID": "UUID",
    "HUGEINT": "NUMERIC(38,0)",
    "INTERVAL": "INTERVAL",
}

# DuckDB -> StarRocks base type mapping
_DUCKDB_TO_STARROCKS = {
    "VARCHAR": "VARCHAR(65533)",
    "TEXT": "STRING",
    "STRING": "STRING",
    "CHAR": "CHAR",
    "INTEGER": "INT",
    "INT": "INT",
    "INT4": "INT",
    "BIGINT": "BIGINT",
    "INT8": "BIGINT",
    "SMALLINT": "SMALLINT",
    "INT2": "SMALLINT",
    "TINYINT": "TINYINT",
    "DOUBLE": "DOUBLE",
    "FLOAT8": "DOUBLE",
    "FLOAT": "FLOAT",
    "FLOAT4": "FLOAT",
    "REAL": "FLOAT",
    "DECIMAL": "DECIMAL(38,9)",
    "NUMERIC": "DECIMAL(38,9)",
    "BOOLEAN": "BOOLEAN",
    "BOOL": "BOOLEAN",
    "DATE": "DATE",
    "TIMESTAMP": "DATETIME",
    "TIMESTAMP WITH TIME ZONE": "DATETIME",
    "TIMESTAMPTZ": "DATETIME",
    "TIME": "VARCHAR(20)",
    "UUID": "VARCHAR(36)",
    "HUGEINT": "LARGEINT",
    "INTERVAL": "VARCHAR(50)",
}

_DIALECT_MAPS = {
    ("duckdb", "greenplum"): _DUCKDB_TO_GREENPLUM,
    ("duckdb", "starrocks"): _DUCKDB_TO_STARROCKS,
    ("duckdb", "postgresql"): _DUCKDB_TO_GREENPLUM,
}


class UnsupportedTypeError(Exception):
    """Raised when a column type cannot be mapped to the target dialect."""

    def __init__(self, column_name: str, source_type: str, target_dialect: str = ""):
        self.column_name = column_name
        self.source_type = source_type
        self.target_dialect = target_dialect
        super().__init__(
            f"Column '{column_name}' has unsupported type '{source_type}'"
            f"{f' for target dialect {target_dialect}' if target_dialect else ''}"
        )


def _parse_type(raw_type: str) -> tuple:
    """Parse a type string into (base_type, params).

    Examples:
        "DECIMAL(10,2)" -> ("DECIMAL", "(10,2)")
        "VARCHAR"       -> ("VARCHAR", "")
        "TIMESTAMP WITH TIME ZONE" -> ("TIMESTAMP WITH TIME ZONE", "")
    """
    raw_type = raw_type.strip()
    match = _TYPE_PARAM_RE.match(raw_type)
    if not match:
        return raw_type.upper(), ""
    base = match.group(1).strip().upper()
    params = match.group(2) or ""
    return base, params


def map_columns_between_dialects(
    columns: List[dict],
    source_dialect: str,
    target_dialect: str,
    target_profile: Optional[str] = None,
) -> List[dict]:
    """Map column types from source dialect to target dialect.

    Args:
        columns: List of column definitions, each with keys: name, type, nullable.
        source_dialect: Source database dialect (e.g. "duckdb").
        target_dialect: Target database dialect (e.g. "greenplum", "starrocks").
        target_profile: Optional profile name for additional mapping context.

    Returns:
        List of column dicts with an added 'target_type' field.

    Raises:
        UnsupportedTypeError: If a column type cannot be mapped.
        ValueError: If the dialect pair is not supported.
    """
    if not columns:
        return []

    key = (source_dialect.lower(), target_dialect.lower())
    type_map = _DIALECT_MAPS.get(key)
    if type_map is None:
        raise ValueError(f"Unsupported dialect pair: {source_dialect} -> {target_dialect}")

    result = []
    for col in columns:
        mapped = dict(col)  # shallow copy
        base_type, params = _parse_type(col["type"])

        # Check unsupported types
        if base_type in _UNSUPPORTED_TYPES:
            raise UnsupportedTypeError(col["name"], col["type"].strip(), target_dialect)

        target_type = type_map.get(base_type)
        if target_type is None:
            raise UnsupportedTypeError(col["name"], col["type"].strip(), target_dialect)

        # Parameter resolution logic:
        # - Source has params (e.g. DECIMAL(10,2)): always use source params
        # - Source has no params but target has defaults (e.g. VARCHAR -> VARCHAR(65533)): use target default
        # - Neither has params: use target base type as-is
        if params:
            # Source has explicit params — strip any default params from target and use source params
            target_base = re.sub(r"\(.*\)", "", target_type).strip()
            mapped["target_type"] = target_base + params
        else:
            mapped["target_type"] = target_type

        result.append(mapped)

    return result
