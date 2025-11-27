# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot import expressions
from sqlglot.expressions import CTE, Table

from datus.utils.constants import DBType, SQLType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def parse_read_dialect(dialect: str = DBType.SNOWFLAKE) -> str:
    """Map SQL dialect to the appropriate read dialect for sqlglot parsing."""
    db = (dialect or "").strip().lower()
    if db in (DBType.POSTGRES, DBType.POSTGRESQL, "redshift", "greenplum"):
        return DBType.POSTGRES
    if db in ("spark", "databricks", DBType.HIVE, DBType.STARROCKS):
        return DBType.HIVE
    if db in (DBType.MSSQL, DBType.SQLSERVER):
        return "tsql"
    return dialect


def parse_dialect(dialect: str = DBType.SNOWFLAKE) -> str:
    """Map SQL dialect to the dialect for sqlglot parsing."""
    db = (dialect or "").strip().lower()
    if db in (DBType.POSTGRES, DBType.POSTGRESQL):
        return DBType.POSTGRES
    if db in (DBType.MSSQL, DBType.SQLSERVER):
        return "tsql"
    return dialect


def parse_metadata_from_ddl(sql: str, dialect: str = DBType.SNOWFLAKE) -> Dict[str, Any]:
    """
    Parse SQL CREATE TABLE statement and return structured table and column information.

    Args:
        sql: SQL CREATE TABLE statement
        dialect: SQL dialect (mysql, oracle, postgre, snowflake, bigquery...)

    Returns:
        Dict containing:
        {
            "table": {
                "name": str,
                "comment": str
            },
            "columns": [
                {
                    "name": str,
                    "type": str,
                    "comment": str
                }
            ]
        }
    """
    dialect = parse_dialect(dialect)

    try:
        result = {"table": {"name": "", "schema_name": "", "database_name": ""}, "columns": []}

        # Parse SQL using sqlglot with error handling
        parsed = sqlglot.parse_one(sql.strip(), dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)

        if isinstance(parsed, sqlglot.exp.Create):
            tb_info = parsed.find_all(Table).__next__()
            # Get table name
            table_name = tb_info.name

            if isinstance(table_name, str):
                table_name = table_name.strip('"').strip("`").strip("[]")
            result["table"]["name"] = table_name
            result["table"]["schema_name"] = tb_info.db
            result["table"]["database_name"] = tb_info.catalog
            if tb_info.comments:
                result["table"]["comment"] = tb_info.comments

            # Get column definitions
            for column in parsed.this.expressions:
                if isinstance(column, sqlglot.exp.ColumnDef):
                    col_name = column.name
                    if isinstance(col_name, str):
                        col_name = col_name.strip('"').strip("`").strip("[]")

                    col_dict = {"name": col_name, "type": str(column.kind)}

                    # Get column comment if exists
                    if hasattr(column, "comments") and column.comments:
                        col_dict["comment"] = column.comments
                    elif hasattr(column, "comment") and column.comment:
                        col_dict["comment"] = column.comment

                    result["columns"].append(col_dict)

        return result

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {"table": {"name": ""}, "columns": []}


def extract_table_names(sql, dialect=DBType.SNOWFLAKE, ignore_empty=False) -> List[str]:
    """
    Extract fully qualified table names (database.schema.table) from SQL.
    Returns a list of unique table names with original case preserved.
    Filters out CTE (Common Table Expression) tables.
    """
    # Parse the SQL using sqlglot
    read_dialect = parse_read_dialect(dialect)
    try:
        parsed = sqlglot.parse_one(sql, read=read_dialect, error_level=sqlglot.ErrorLevel.IGNORE)
        if parsed is None:
            return []
    except Exception as e:
        logger.warning(f"Error parsing SQL {sql}, error: {e}")
        return []
    table_names = []

    # Get all CTE names
    cte_names = set()
    for cte in parsed.find_all(CTE):
        if hasattr(cte, "alias") and cte.alias:
            cte_names.add(cte.alias.lower())

    for tb in parsed.find_all(Table):
        db = tb.catalog
        schema = tb.db
        table_name = tb.name

        # Skip if the table is a CTE
        if table_name.lower() in cte_names:
            continue
        full_name = []

        if dialect in [DBType.MYSQL, DBType.ORACLE, DBType.POSTGRES, DBType.POSTGRESQL]:
            if not ignore_empty or schema:
                full_name.append(schema)
        elif dialect not in (DBType.SQLITE,):
            if not ignore_empty or db:
                full_name.append(db)
            if not ignore_empty or schema:
                full_name.append(schema)
        full_name.append(table_name)

        table_names.append(".".join(full_name))

    return list(set(table_names))  # Remove duplicates


def metadata_identifier(
    catalog_name: str = "",
    database_name: str = "",
    schema_name: str = "",
    table_name: str = "",
    dialect: str = DBType.SNOWFLAKE,
) -> str:
    """
    Generate a unique identifier for a table based on its metadata.
    """
    if dialect == DBType.SQLITE:
        return f"{database_name}.{table_name}" if database_name else table_name
    elif dialect == DBType.DUCKDB:
        return f"{database_name}.{schema_name}.{table_name}"
    elif dialect in (DBType.MYSQL, DBType.STARROCKS):
        return f"{catalog_name}.{database_name}.{table_name}" if catalog_name else f"{database_name}.{table_name}"
    elif dialect in (DBType.ORACLE, DBType.POSTGRESQL, DBType.POSTGRES):
        return f"{database_name}.{schema_name}.{table_name}"
    elif dialect == DBType.SNOWFLAKE:
        return (
            f"{catalog_name}.{database_name}.{schema_name}.{table_name}"
            if catalog_name
            else f"{database_name}.{schema_name}.{table_name}"
        )
    elif dialect == "databricks":
        return f"{catalog_name}.{schema_name}.{table_name}" if catalog_name else f"{schema_name}.{table_name}"
    return table_name


def parse_table_name_parts(full_table_name: str, dialect: str = DBType.SNOWFLAKE) -> Dict[str, str]:
    """
    Parse a full table name into its component parts (catalog, database, schema, table).

    Args:
        full_table_name: Full table name string (e.g., "database.schema.table")
        dialect: SQL dialect to determine parsing logic

    Returns:
        Dict with keys: catalog_name, database_name, schema_name, table_name

    Examples:
        For DuckDB:
        - "table" -> {"catalog_name": "", "database_name": "", "schema_name": "", "table_name": "table"}
        - "schema.table" -> {"catalog_name": "", "database_name": "", "schema_name": "schema", "table_name": "table"}
        - "database.schema.table" -> {"catalog_name": "", "database_name": "database",
                                      "schema_name": "schema", "table_name": "table"}
    """
    # Database-specific field mapping configurations
    # Each list represents the field order from left to right in the table name
    DB_FIELD_MAPPINGS = {
        DBType.DUCKDB.value: ["database_name", "schema_name", "table_name"],  # max 3 parts
        DBType.SQLITE.value: ["database_name", "table_name"],  # max 2 parts
        DBType.STARROCKS.value: ["catalog_name", "database_name", "table_name"],  # max 3 parts, no schema
        DBType.SNOWFLAKE.value: ["catalog_name", "database_name", "schema_name", "table_name"],  # max 4 parts
    }

    dialect = parse_dialect(dialect)

    # Split the table name by dots
    # Handle different quote styles: `backticks`, "double quotes", [brackets]
    quote_patterns = [
        r'(["`])(?:(?=(\\?))\2.)*?\1',  # "quoted" or `quoted`
        r"\[(.*?)\]",  # [bracketed]
    ]

    # Find all quoted parts
    parts = []

    # First, extract all quoted parts
    for pattern in quote_patterns:
        matches = re.findall(pattern, full_table_name)
        if matches:
            # Handle different regex return formats
            if isinstance(matches[0], tuple):
                # Pattern returns tuples, extract the actual content
                for match in matches:
                    if isinstance(match, tuple):
                        part = match[0] if match[0] else match[1] if len(match) > 1 else ""
                    else:
                        part = str(match)
                    if part and part not in parts:
                        parts.append(part.strip('"`[]'))
            else:
                # Pattern returns strings
                parts.extend([str(m).strip('"`[]') for m in matches])

    # If no quoted parts found, split by dots
    if not parts:
        parts = [part.strip() for part in full_table_name.split(".")]
    else:
        # Split by dots, but respect quotes
        pattern = r'(?:["`\[][^"`\]]*["`\]]|[^.])+'
        matches = re.findall(pattern, full_table_name)
        parts = [match.strip('"`[] ') for match in matches]

    # Clean up parts - remove empty strings
    parts = [p for p in parts if p]

    # Initialize result with empty strings
    result = {"catalog_name": "", "database_name": "", "schema_name": "", "table_name": ""}

    # Get field mapping for the dialect, or use default mapping
    if dialect in DB_FIELD_MAPPINGS:
        field_mapping = DB_FIELD_MAPPINGS[dialect]
        max_parts = len(field_mapping)

        # If we have more parts than expected, take the last N parts
        if len(parts) > max_parts:
            parts = parts[-max_parts:]

        # Map parts to fields according to the configuration
        # We map from right to left (table_name is always the last part)
        for i, part in enumerate(reversed(parts)):
            if i < len(field_mapping):
                field_name = field_mapping[-(i + 1)]  # Get field name from right to left
                result[field_name] = part
    else:
        # Default behavior for unknown dialects: assume last part is table name
        result["table_name"] = parts[-1]
        if len(parts) > 1:
            result["schema_name"] = parts[-2]
        if len(parts) > 2:
            result["database_name"] = parts[-3]
        if len(parts) > 3:
            result["catalog_name"] = parts[-4]

    return result


def parse_table_names_parts(full_table_names: List[str], dialect: str = DBType.SNOWFLAKE) -> List[Dict[str, str]]:
    """
    Parse a list of full table names into their component parts.

    Args:
        full_table_names: List of full table name strings
        dialect: SQL dialect to determine parsing logic

    Returns:
        List of dicts with keys: catalog_name, database_name, schema_name, table_name
    """
    return [parse_table_name_parts(table_name, dialect) for table_name in full_table_names]


_METADATA_RE: re.Pattern | None = None


def _metadata_pattern() -> re.Pattern:
    global _METADATA_RE
    if not _METADATA_RE:
        _METADATA_RE = re.compile(
            r"""(?ix)^\s*
        (?:
            show\b(?:\s+create\s+table|\s+catalogs|\s+databases|\s+tables|\s+functions|\s+views|\s+columns|\s+partitions)?
            |set\s+catalog\b
            |describe\b
            |pragma\b
        )
    """,
        )
    return _METADATA_RE


def strip_sql_comments(sql: str) -> str:
    """Remove /* ... */ and -- ... comments (simple but effective)."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
    return sql


def _is_escaped(text: str, index: int) -> bool:
    """Return True if the character at index is escaped by an odd number of backslashes."""
    backslash_count = 0
    position = index - 1
    while position >= 0 and text[position] == "\\":
        backslash_count += 1
        position -= 1
    return backslash_count % 2 == 1


_DOLLAR_QUOTE_RE = re.compile(r"\$[A-Za-z_0-9]*\$")


def _match_dollar_tag(text: str, index: int) -> Optional[str]:
    """Return the dollar-quote tag starting at index, if any."""
    match = _DOLLAR_QUOTE_RE.match(text, index)
    if not match:
        return None
    return match.group(0)


def _first_statement(sql: str) -> str:
    """Return the first non-empty statement (before the first ';'), with comments removed."""
    s = strip_sql_comments(sql).strip()
    if not s:
        return ""

    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    in_bracket = False
    dollar_tag: Optional[str] = None

    i = 0
    length = len(s)
    while i < length:
        ch = s[i]

        if dollar_tag:
            if s.startswith(dollar_tag, i):
                i += len(dollar_tag)
                dollar_tag = None
                continue
            i += 1
            continue

        if in_single_quote:
            if ch == "'":
                if i + 1 < length and s[i + 1] == "'":
                    i += 2
                    continue
                if not _is_escaped(s, i):
                    in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            if ch == '"':
                if i + 1 < length and s[i + 1] == '"':
                    i += 2
                    continue
                if not _is_escaped(s, i):
                    in_double_quote = False
            i += 1
            continue

        if in_backtick:
            if ch == "`":
                if i + 1 < length and s[i + 1] == "`":
                    i += 2
                    continue
                in_backtick = False
            i += 1
            continue

        if in_bracket:
            if ch == "]":
                in_bracket = False
            i += 1
            continue

        # Not within any quote context
        if ch == "'":
            in_single_quote = True
            i += 1
            continue
        if ch == '"':
            in_double_quote = True
            i += 1
            continue
        if ch == "`":
            in_backtick = True
            i += 1
            continue
        if ch == "[":
            in_bracket = True
            i += 1
            continue
        if ch == "$":
            tag = _match_dollar_tag(s, i)
            if tag:
                dollar_tag = tag
                i += len(tag)
                continue

        if ch == ";":
            return s[:i].strip()

        i += 1

    return s.strip()


_KEYWORD_SQL_TYPE_MAP: Dict[str, SQLType] = {
    "SELECT": SQLType.SELECT,
    "VALUES": SQLType.SELECT,
    "WITH": SQLType.SELECT,
    "INSERT": SQLType.INSERT,
    "REPLACE": SQLType.INSERT,
    "UPDATE": SQLType.UPDATE,
    "DELETE": SQLType.DELETE,
    "MERGE": SQLType.MERGE,
    "CREATE": SQLType.DDL,
    "ALTER": SQLType.DDL,
    "DROP": SQLType.DDL,
    "TRUNCATE": SQLType.DDL,
    "RENAME": SQLType.DDL,
    "COMMENT": SQLType.DDL,
    "GRANT": SQLType.DDL,
    "REVOKE": SQLType.DDL,
    "ANALYZE": SQLType.DDL,
    "VACUUM": SQLType.DDL,
    "OPTIMIZE": SQLType.DDL,
    "COPY": SQLType.DDL,
    "REFRESH": SQLType.DDL,
    "SHOW": SQLType.METADATA_SHOW,
    "DESCRIBE": SQLType.METADATA_SHOW,
    "DESC": SQLType.METADATA_SHOW,
    "PRAGMA": SQLType.METADATA_SHOW,
    "EXPLAIN": SQLType.EXPLAIN,
    "USE": SQLType.CONTENT_SET,
    "SET": SQLType.CONTENT_SET,
    "CALL": SQLType.CONTENT_SET,
    "EXEC": SQLType.CONTENT_SET,
    "EXECUTE": SQLType.CONTENT_SET,
    "BEGIN": SQLType.CONTENT_SET,
    "START": SQLType.CONTENT_SET,
    "COMMIT": SQLType.CONTENT_SET,
    "ROLLBACK": SQLType.CONTENT_SET,
}

_OPTIONAL_DDL_EXPRESSIONS: tuple[type[expressions.Expression], ...] = tuple(
    getattr(expressions, name)
    for name in (
        "Copy",
        "Refresh",
    )
    if hasattr(expressions, name)
)


def _normalize_expression(expr: Optional[expressions.Expression]) -> Optional[expressions.Expression]:
    """
    Unwrap container expressions (Alias, Subquery, Paren) to reach the semantic root expression.
    """
    while expr is not None and isinstance(expr, (expressions.Alias, expressions.Subquery, expressions.Paren)):
        expr = expr.this
    return expr


def _fallback_sql_type(statement: str) -> SQLType | None:
    """Infer the SQL type from leading keywords when parsing fails."""
    if not statement:
        return None

    upper_stmt = statement.upper()
    match = re.match(r"\s*([A-Z_]+)", upper_stmt)
    keyword = match.group(1) if match else ""

    if keyword == "WITH":
        # Look for the statement keyword that follows all CTE definitions.
        match_cte_target = re.search(r"\)\s*(SELECT|INSERT|UPDATE|DELETE|MERGE)\b", upper_stmt)
        if match_cte_target:
            keyword = match_cte_target.group(1)
        else:
            keyword = "SELECT"

    if not keyword:
        return None

    return _KEYWORD_SQL_TYPE_MAP.get(keyword)


def parse_sql_type(sql: str, dialect: str) -> SQLType:
    """
    Determines the type of an SQL statement based on its first keyword.

    This function analyzes the beginning of an SQL query to classify it into
    one of the SQLType categories (SELECT, DDL, METADATA, etc.). It is designed
    to handle common SQL commands across different database dialects.

    Args:
        sql: The SQL query string.
        dialect: SQL dialect to determine parsing logic

    Returns:
        The determined SQLType enum member. Returns SQLType.UNKNOWN if parsing fails.
    """
    if not sql or not isinstance(sql, str):
        return SQLType.UNKNOWN

    stripped_sql = sql.strip()
    if not stripped_sql:
        return SQLType.UNKNOWN

    first_statement = _first_statement(stripped_sql)
    dialect_name = parse_dialect(dialect)
    try:
        parsed_expression = sqlglot.parse_one(
            first_statement, dialect=dialect_name, error_level=sqlglot.ErrorLevel.IGNORE
        )
        if parsed_expression is None:
            if dialect_name == DBType.STARROCKS.value and _metadata_pattern().match(first_statement):
                return SQLType.METADATA_SHOW
            inferred = _fallback_sql_type(first_statement)
            return inferred if inferred else SQLType.UNKNOWN
    except Exception:
        inferred = _fallback_sql_type(first_statement)
        return inferred if inferred else SQLType.UNKNOWN

    normalized_expression = _normalize_expression(parsed_expression)
    if isinstance(normalized_expression, expressions.Query):
        return SQLType.SELECT
    if isinstance(normalized_expression, expressions.Values):
        return SQLType.SELECT
    if isinstance(normalized_expression, expressions.Insert):
        return SQLType.INSERT
    if isinstance(normalized_expression, expressions.Merge):
        return SQLType.MERGE
    if isinstance(normalized_expression, expressions.Update):
        return SQLType.UPDATE
    if isinstance(normalized_expression, expressions.Delete):
        return SQLType.DELETE
    if isinstance(
        normalized_expression,
        (
            expressions.Create,
            expressions.Alter,
            expressions.Drop,
            expressions.TruncateTable,
            expressions.RenameColumn,
            expressions.Analyze,
            expressions.Comment,
            expressions.Grant,
        ),
    ):
        return SQLType.DDL
    if isinstance(normalized_expression, (expressions.Describe, expressions.Show, expressions.Pragma)):
        return SQLType.METADATA_SHOW
    if isinstance(normalized_expression, expressions.Command):
        command_name = str(normalized_expression.args.get("this") or "").upper()
        if command_name in {"SHOW", "DESC", "DESCRIBE"}:
            return SQLType.METADATA_SHOW
        if command_name == "EXPLAIN":
            return SQLType.EXPLAIN
        if command_name == "REPLACE":
            return SQLType.INSERT
        if command_name in {"CALL", "EXEC", "EXECUTE"}:
            return SQLType.CONTENT_SET
        return SQLType.CONTENT_SET
    if isinstance(
        normalized_expression,
        (
            expressions.Use,
            expressions.Transaction,
            expressions.Commit,
            expressions.Rollback,
            expressions.Set,
        ),
    ):
        return SQLType.CONTENT_SET
    if _OPTIONAL_DDL_EXPRESSIONS and isinstance(normalized_expression, _OPTIONAL_DDL_EXPRESSIONS):
        return SQLType.DDL

    inferred = _fallback_sql_type(first_statement)
    return inferred if inferred else SQLType.UNKNOWN


_CONTEXT_CMD_RE = re.compile(r"^\s*(use|set)\b", flags=re.IGNORECASE)


def _identifier_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, expressions.Identifier):
        return value.name
    if isinstance(value, expressions.Literal):
        literal = value.this
        return literal if isinstance(literal, str) else str(literal)
    if isinstance(value, expressions.Table):
        return _identifier_name(value.this)
    if isinstance(value, expressions.Expression):
        return value.sql()
    if isinstance(value, str):
        return value.strip('"`[]')
    return str(value)


def _table_parts(table_expr: Optional[Table]) -> Dict[str, str]:
    if not isinstance(table_expr, Table):
        return {"catalog": "", "database": "", "identifier": ""}
    args = table_expr.args
    return {
        "catalog": _identifier_name(args.get("catalog")),
        "database": _identifier_name(args.get("db")),
        "identifier": _identifier_name(args.get("this")),
    }


def _parse_identifier_sequence(value: str, dialect: str) -> Dict[str, str]:
    parsed = sqlglot.parse_one(f"USE {value}", dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)
    table_expr = parsed.this if isinstance(parsed, expressions.Use) else None
    return _table_parts(table_expr)


def parse_context_switch(sql: str, dialect: str) -> Optional[Dict[str, Any]]:
    """
    Parse statements that switch catalog/database/schema context (USE/SET).

    Returns a dict with keys:
        command: The leading verb ("USE" or "SET")
        target:  The logical object being switched ("catalog", "database", "schema")
        catalog_name, database_name, schema_name: Extracted identifiers (empty string if absent)
        fuzzy: Whether the target inference is best-effort (e.g., DuckDB bare USE)
        raw: The first statement that was parsed
    """
    if not sql or not isinstance(sql, str):
        return None

    statement = _first_statement(sql)
    if not statement:
        return None

    cmd_match = _CONTEXT_CMD_RE.match(statement)
    if not cmd_match:
        return None

    command = cmd_match.group(1).upper()
    normalized_dialect = parse_dialect(dialect)

    result: Dict[str, Any] = {
        "command": command,
        "target": "",
        "catalog_name": "",
        "database_name": "",
        "schema_name": "",
        "fuzzy": False,
        "raw": statement,
    }

    if command == "USE":
        expression = sqlglot.parse_one(statement, dialect=normalized_dialect, error_level=sqlglot.ErrorLevel.IGNORE)
        if not isinstance(expression, expressions.Use):
            return None
        parts = _table_parts(expression.this)
        kind_expr = expression.args.get("kind")
        kind = kind_expr.name.upper() if isinstance(kind_expr, expressions.Var) else ""

        catalog = parts["catalog"]
        database = parts["database"]
        identifier = parts["identifier"]

        if not identifier and not database and not catalog:
            return None

        if kind == "CATALOG":
            result["catalog_name"] = identifier or database or catalog
            result["target"] = "catalog"
            return result

        if kind == "DATABASE":
            result["database_name"] = identifier or database
            result["target"] = "database"
            return result

        if kind == "SCHEMA":
            result["schema_name"] = identifier
            if catalog:
                result["catalog_name"] = catalog
            if database:
                result["database_name"] = database
            result["target"] = "schema"
            return result

        # Dialect-specific fallbacks when the kind keyword is omitted
        if normalized_dialect == DBType.DUCKDB.value:
            if database:
                result["database_name"] = database
                result["schema_name"] = identifier
                result["target"] = "schema"
            else:
                result["schema_name"] = identifier
                result["target"] = "schema"
                result["fuzzy"] = True
            return result

        if normalized_dialect == DBType.MYSQL.value:
            result["database_name"] = identifier
            result["target"] = "database"
            return result

        if normalized_dialect == DBType.STARROCKS.value:
            if catalog or (database and not catalog):
                result["catalog_name"] = catalog or database
                result["database_name"] = identifier
            else:
                result["database_name"] = identifier
            result["target"] = "database"
            return result

        if normalized_dialect == DBType.SNOWFLAKE.value:
            if catalog:
                result["catalog_name"] = catalog
                result["database_name"] = database
                result["schema_name"] = identifier
                result["target"] = "schema"
            elif database:
                result["database_name"] = database
                result["schema_name"] = identifier
                result["target"] = "schema"
            else:
                result["database_name"] = identifier
                result["target"] = "database"
            return result

        # Generic fallback
        if catalog:
            result["catalog_name"] = catalog
        if database:
            result["database_name"] = database
        result["schema_name"] = identifier
        result["target"] = "schema" if database or catalog else "database"
        return result

    if command == "SET":
        set_match = re.match(
            r"^\s*SET\s+(?:SESSION\s+)?(CATALOG|DATABASE|SCHEMA)\s+(.*)$", statement, flags=re.IGNORECASE
        )
        if not set_match:
            return None

        target = set_match.group(1).upper()
        remainder = set_match.group(2).strip()
        remainder = remainder.rstrip(";").strip()
        if remainder.startswith("="):
            remainder = remainder[1:].strip()
        elif remainder.upper().startswith("TO "):
            remainder = remainder[3:].strip()

        if not remainder:
            return None

        parts = _parse_identifier_sequence(remainder, normalized_dialect)
        catalog = parts["catalog"]
        database = parts["database"]
        identifier = parts["identifier"]

        if target == "CATALOG":
            result["target"] = "catalog"
            result["catalog_name"] = identifier or database or catalog
            return result

        if target == "DATABASE":
            result["target"] = "database"
            result["catalog_name"] = catalog
            result["database_name"] = identifier or database
            return result

        if target == "SCHEMA":
            result["target"] = "schema"
            result["catalog_name"] = catalog
            result["database_name"] = database
            result["schema_name"] = identifier
            if normalized_dialect == DBType.DUCKDB.value and not database:
                # DuckDB SET SCHEMA mirrors USE without database context.
                result["fuzzy"] = False
            return result

    return None


def normalize_sql(sql: str) -> str:
    # 1) Replace all line breaks and tabs with a space
    s = re.sub(r"[\r\n\t]+", " ", sql)
    # 2) Shrink multiple spaces into a single space
    s = re.sub(r" +", " ", s)
    # 3) Remove the spaces at both ends
    s = s.strip()
    return s


def format_sql_to_pretty(sql: str, dialect: str) -> str:
    """Pretty print SQL if possible, otherwise return the original text."""
    if not sql:
        return sql
    read_dialect = parse_read_dialect(dialect)
    try:
        formatted = sqlglot.transpile(sql, read=read_dialect, pretty=True)
        if formatted:
            return formatted[0]
    except Exception as exc:
        logger.debug(f"Failed to format SQL for download: {exc}")
    return sql
