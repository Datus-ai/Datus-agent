# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot import expressions
from sqlglot.expressions import CTE, Table

from datus.utils.config_utils import coerce_bool
from datus.utils.constants import DBType, SQLType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _registry_hook(name: str, dialect: str):
    from datus.tools.db_tools import connector_registry

    getter = getattr(connector_registry, name, None)
    return getter(dialect) if callable(getter) else None


def parse_read_dialect(dialect: str = "snowflake") -> str:
    """Map SQL dialect to the appropriate read dialect for sqlglot parsing."""
    db = (dialect or "").strip().lower()
    registered = _registry_hook("get_parser_dialect", db)
    if registered:
        return registered
    if db in ("postgres", "postgresql", "redshift", "greenplum"):
        return "postgres"
    if db in ("spark", "databricks", "hive", "starrocks"):
        return "hive"
    if db in ("mssql", "sqlserver"):
        return "tsql"
    return dialect


def parse_dialect(dialect: str = "snowflake") -> str:
    """Map SQL dialect to the dialect for sqlglot parsing."""
    db = (dialect or "").strip().lower()
    registered = _registry_hook("get_parser_dialect", db)
    if registered:
        return registered
    if db in ("postgres", "postgresql"):
        return "postgres"
    if db in ("mssql", "sqlserver"):
        return "tsql"
    return dialect


def parse_metadata_from_ddl(sql: str, dialect: str = "snowflake") -> Dict[str, Any]:
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
    raw_dialect = (dialect or "").strip().lower()
    dialect = parse_dialect(raw_dialect)

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
            identifier_parser = _registry_hook("get_identifier_parser", raw_dialect)
            if identifier_parser:
                parsed_name = identifier_parser(
                    ".".join(part for part in (tb_info.catalog, tb_info.db, table_name) if part)
                )
                result["table"]["schema_name"] = parsed_name["schema_name"]
                result["table"]["database_name"] = parsed_name["database_name"]
            else:
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


def extract_table_names(sql, dialect="snowflake", ignore_empty=False) -> List[str]:
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
    return _table_names_from_expression(parsed, dialect=dialect, ignore_empty=ignore_empty)


def extract_table_names_strict(sql, dialect="snowflake", ignore_empty=False) -> List[str]:
    """Extract table names and raise when sqlglot cannot parse the SQL."""
    read_dialect = parse_read_dialect(dialect)
    parsed = sqlglot.parse_one(sql, read=read_dialect, error_level=sqlglot.ErrorLevel.RAISE)
    if parsed is None:
        raise ValueError("SQL parser returned no expression")
    return _table_names_from_expression(parsed, dialect=dialect, ignore_empty=ignore_empty)


def _table_names_from_expression(parsed, *, dialect: str, ignore_empty: bool) -> List[str]:
    """Collect physical table names from an already parsed sqlglot expression."""
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
        if _registry_hook("get_identifier_parser", dialect):
            parts = []
            if not ignore_empty or db:
                parts.append(db)
            if not ignore_empty or schema:
                parts.append(schema)
            parts.append(table_name)
            table_names.append(".".join(parts))
            continue

        full_name = []

        if dialect in ["mysql", "oracle", "postgres", "postgresql"]:
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
    dialect: str = "snowflake",
) -> str:
    """
    Generate a unique identifier for a table based on its metadata.
    """
    from datus.tools.db_tools import connector_registry

    # Built-in connectors
    if dialect == DBType.SQLITE:
        return f"{database_name}.{table_name}" if database_name else table_name
    if dialect == DBType.DUCKDB:
        return f"{database_name}.{schema_name}.{table_name}"
    # External dialects: build identifier from registry capabilities
    parts = []
    if connector_registry.support_catalog(dialect) and catalog_name:
        parts.append(catalog_name)
    if connector_registry.support_database(dialect) and database_name:
        parts.append(database_name)
    if connector_registry.support_schema(dialect) and schema_name:
        parts.append(schema_name)
    parts.append(table_name)
    return ".".join(parts)


#: Widest identifier shape, in right-align order. Used for dialects that declare
#: no namespace capabilities at all (adapter not installed, or one that reports
#: none): assuming the widest shape keeps a qualified prefix in a namespace field
#: instead of collapsing it into ``table_name``.
_WIDEST_FIELD_ORDER = ("catalog_name", "database_name", "schema_name", "table_name")


def table_name_field_order(dialect: str = "snowflake") -> List[str]:
    """Return the fields a dotted identifier right-aligns into for *dialect*.

    Single source of truth for the identifier *shape*, shared by
    :func:`parse_table_name_parts` (which right-aligns onto this order) and by
    callers that need to **emit** an identifier or scope token with a matching
    segment count. Emitting fewer segments than the order has silently lands
    every literal in the wrong field: ``default_catalog.*`` on StarRocks
    (``[catalog, database, table]``) parses as ``{database: default_catalog,
    table: *}`` and matches no real table.

    Adapters that ship their own ``get_identifier_parser`` are expected to
    right-align the same way; their order still follows the capabilities they
    declare.
    """
    d = parse_dialect((dialect or "").strip().lower())
    from datus.tools.db_tools import connector_registry

    # Built-in connectors
    if d == DBType.SQLITE:
        return ["database_name", "table_name"]
    if d == DBType.DUCKDB:
        return ["database_name", "schema_name", "table_name"]
    # External dialects: derive from registry
    fields = []
    if connector_registry.support_catalog(d):
        fields.append("catalog_name")
    if connector_registry.support_database(d):
        fields.append("database_name")
    if connector_registry.support_schema(d):
        fields.append("schema_name")
    if not fields:
        return list(_WIDEST_FIELD_ORDER)
    fields.append("table_name")
    return fields


def parse_table_name_parts(full_table_name: str, dialect: str = "snowflake") -> Dict[str, str]:
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
    raw_dialect = (dialect or "").strip().lower()
    identifier_parser = _registry_hook("get_identifier_parser", raw_dialect)
    if identifier_parser:
        parsed = identifier_parser(full_table_name)
        expected_fields = {"catalog_name", "database_name", "schema_name", "table_name"}
        if not isinstance(parsed, dict) or not expected_fields.issubset(parsed):
            raise ValueError(
                "Adapter identifier parser must return catalog_name, database_name, schema_name, and table_name"
            )
        return {field: str(parsed[field] or "") for field in expected_fields}

    dialect = parse_dialect(raw_dialect)

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

    if not parts:
        return result

    # Right-align the parts onto the dialect's field order (always >= 2 fields,
    # so an unknown dialect degrades to the widest shape rather than to a bare
    # table name).
    field_mapping = table_name_field_order(dialect)
    max_parts = len(field_mapping)

    # If we have more parts than expected, take the last N parts
    if len(parts) > max_parts:
        parts = parts[-max_parts:]

    # Map parts to fields according to the configuration
    # We map from right to left (table_name is always the last part)
    for i, part in enumerate(reversed(parts)):
        field_name = field_mapping[-(i + 1)]  # Get field name from right to left
        result[field_name] = part

    return result


def parse_table_names_parts(full_table_names: List[str], dialect: str = "snowflake") -> List[Dict[str, str]]:
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
    """Remove ``/* ... */`` and ``-- ...`` comments.

    Scanned rather than regexed, because a comment marker inside a string
    literal is not a comment: ``re.sub`` on ``SELECT 'a--b' FROM t`` cuts the
    statement down to ``SELECT 'a`` and hands that to whoever asked. Several
    callers execute the stripped text, so the truncation reaches the database.
    """
    out: List[str] = []
    i = 0
    length = len(sql)
    while i < length:
        ch = sql[i]

        # A quoted run is copied through whole; nothing inside it is a marker.
        closer = {"'": "'", '"': '"', "`": "`", "[": "]"}.get(ch)
        if closer:
            out.append(ch)
            i += 1
            while i < length:
                out.append(sql[i])
                if sql[i] == closer:
                    # A doubled quote is an escaped one, not the end.
                    if i + 1 < length and sql[i + 1] == closer and closer != "]":
                        out.append(sql[i + 1])
                        i += 2
                        continue
                    if closer == "]" or not _is_escaped(sql, i):
                        i += 1
                        break
                i += 1
            continue

        if ch == "$":
            tag = _match_dollar_tag(sql, i)
            if tag:
                end = sql.find(tag, i + len(tag))
                stop = length if end == -1 else end + len(tag)
                out.append(sql[i:stop])
                i = stop
                continue

        if sql.startswith("--", i):
            end = sql.find("\n", i)
            i = length if end == -1 else end
            out.append(" ")
            continue

        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = length if end == -1 else end + 2
            out.append(" ")
            continue

        out.append(ch)
        i += 1

    return "".join(out)


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


def is_read_query_result(result: Any) -> bool:
    """Return True if a ``FuncToolResult.result`` payload is a read-only query
    result set.

    ``execute_sql`` returns the compressor output (a dict carrying
    ``compressed_data``) for SELECT/SHOW/EXPLAIN statements, and a metadata
    payload (``message``/``sql``/...) for INSERT/UPDATE/DELETE/DDL. Consumers
    that only want to seed query context from reads use this to skip writes.
    """
    return isinstance(result, dict) and "compressed_data" in result


def looks_like_sql_file_ref(text: str) -> bool:
    """True if ``text`` is a bare ``.sql`` file reference, not inline SQL.

    ``execute_sql`` accepts either an inline statement or a workspace-relative
    ``.sql`` file path. A real SQL statement always contains whitespace (at
    minimum between the keyword and its operand), so a whitespace-free token
    ending in ``.sql`` is a path. Shared by the execution path (``DBFuncTool``)
    and the permission gate (``PermissionHooks``) so both detect a file
    reference identically.
    """
    s = text.strip()
    return bool(s) and s.endswith(".sql") and "\n" not in s and " " not in s


def read_workspace_sql_file(file_path: str, workspace_root: str) -> str:
    """Read a workspace-relative ``.sql`` file, rejecting unsafe paths.

    Shared by ``DBFuncTool._read_sql_from_file`` (execution) and
    ``PermissionHooks`` (the statement-type gate) so both resolve a ``.sql``
    reference identically. Raises ``ValueError`` on an absolute path, a ``..``
    traversal, or a path escaping the workspace, and ``FileNotFoundError`` when
    the file does not exist.
    """
    import os
    from pathlib import Path

    if os.path.isabs(file_path):
        raise ValueError(f"Absolute paths are not allowed: {file_path}")
    if ".." in file_path:
        raise ValueError(f"Invalid SQL file path: {file_path}")
    root = os.path.expanduser(workspace_root or ".")
    full_path = (Path(root) / file_path).resolve()
    root_resolved = Path(root).resolve()
    if not str(full_path).startswith(str(root_resolved) + os.sep) and full_path != root_resolved:
        raise ValueError(f"SQL file path escapes workspace: {file_path}")
    if not full_path.exists():
        raise FileNotFoundError(file_path)
    return full_path.read_text(encoding="utf-8")


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
            if dialect_name == "starrocks" and _metadata_pattern().match(first_statement):
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


READ_ONLY_MULTI_STATEMENT = "multi_statement"
READ_ONLY_NON_READ = "non_read"
READ_ONLY_WRITABLE_PRAGMA = "writable_pragma"


def write_statement_reads_data(sql: str, dialect: str) -> bool:
    """Whether a write statement reads rows a row policy would have filtered.

    ``CREATE TABLE ... AS SELECT`` is the obvious shape, but it is not the only
    one: ``UPDATE ... FROM``, ``DELETE ... USING`` and ``MERGE ... USING`` all
    name a source table with no ``Select`` node anywhere in the tree, and
    ``RETURNING`` hands the rows straight back to the caller. Each reads rows a
    policy would have withheld and puts them somewhere it does not cover. The
    read policy plugin cannot see any of it — it hooks reads, and these are
    writes — so every surface that can run a write has to ask before running it.

    Anything the parser did not actually understand counts as yes. On a project
    with policies the cost of being wrong in that direction is a refused
    statement; the other direction is a silent copy of the rows the policy
    exists to withhold.
    """
    import sqlglot
    from sqlglot import exp

    try:
        # RAISE, not IGNORE. IGNORE does not raise on syntax it cannot place —
        # it returns an opaque `Command` whose body was never parsed, so
        # "no Select inside" says nothing about the statement. Every unparsed
        # shape has to reach the `except` and be refused.
        #
        # `parse_dialect`, not the connector's own name: a connector reports
        # `postgresql` while sqlglot only knows `postgres`, and handing over the
        # raw value raises for every statement — which this function reads as
        # "contains a read" and refuses every write, plain `CREATE TABLE`
        # included.
        parsed = sqlglot.parse_one(sql, read=parse_dialect(dialect), error_level=sqlglot.ErrorLevel.RAISE)
    except Exception:
        return True
    if parsed is None or isinstance(parsed, exp.Command):
        return True

    # `COPY` parses, but its source is a table reference rather than a Select,
    # and `COPY ... TO '/path'` / `TO PROGRAM` on a self-hosted server is a real
    # export.
    if isinstance(parsed, exp.Copy):
        return True

    # Rows leaving through RETURNING are read by definition.
    if parsed.args.get("returning") or list(parsed.find_all(exp.Returning)):
        return True

    if list(parsed.find_all(exp.Select)):
        return True

    # A source table named without a subquery: `UPDATE t SET .. FROM src`,
    # `DELETE FROM t USING src`, `MERGE INTO t USING src`. Each reads `src`
    # while the tree holds no Select at all.
    #
    # Counted rather than read off a named argument. The argument differs per
    # node and *between sqlglot releases* — an earlier version of this keyed on
    # `Update.args["from_"]`, which passed locally on 30.1 and missed the same
    # statement on CI. `exp.Table` is the one part of the shape that does not
    # move. One DML statement has exactly one target table; a second table in
    # the tree is something being read.
    #
    # Scoped to the DML family on purpose: DDL routinely names two tables
    # without reading a row (`ALTER TABLE a RENAME TO b`, `CREATE TABLE a
    # (LIKE b)`), and the DDL that does read — CTAS — carries a Select and is
    # caught above.
    if isinstance(parsed, (exp.Update, exp.Delete, exp.Merge)):
        if len({table.sql() for table in parsed.find_all(exp.Table)}) > 1:
            return True

    return False


def is_single_statement(sql: str) -> bool:
    """Whether the input is exactly one statement.

    A trailing semicolon and surrounding comments do not make it two. Callers
    that route on statement type need this separately from
    :func:`validate_read_only_sql`, which folds it into one violation code.
    """
    normalized = strip_sql_comments(sql).strip().rstrip(";").strip()
    if not normalized:
        return False
    return _first_statement(normalized) == normalized


#: Leading ``EXPLAIN`` and the option forms the dialects spell it with, so the
#: statement being explained can be classified on its own.
#:
#: Matched textually rather than off the parse tree because the tree is not
#: dependable here: postgres parses ``EXPLAIN ...`` to a ``Command`` whose inner
#: node is a bare string, mysql to a ``Describe`` with a real node, and
#: ``EXPLAIN (ANALYZE, BUFFERS) DELETE ...`` fails to parse on mysql outright.
_EXPLAIN_PREFIX_RE = re.compile(
    r"""^EXPLAIN\b\s*
        (?:
            \([^)]*\)                 # postgres: EXPLAIN (ANALYZE, BUFFERS)
          | ANALYZE\b | ANALYSE\b
          | VERBOSE\b | EXTENDED\b | PARTITIONS\b
          | FORMAT\s*=?\s*\w+
          | QUERY\s+PLAN\b           # sqlite
        )*
        \s*""",
    re.IGNORECASE | re.VERBOSE,
)


def _embeds_write_statement(sql: str, dialect: str) -> bool:
    """Whether ``sql`` performs a write anywhere inside it, not just at the root.

    Postgres data-modifying CTEs are a read on the outside and a write on the
    inside::

        WITH deleted AS (DELETE FROM orders RETURNING *) SELECT * FROM deleted

    That statement's top-level node is a SELECT, so classifying by root alone
    called it a read and let it through every gate — while it deletes the rows.

    Normalizes the dialect the same way ``parse_sql_type`` does and retries
    without one. Passing the raw name meant a dialect sqlglot does not know
    raised ``ValueError``, the walk was skipped, and the CTE above went through
    — the check silently did not apply to exactly the deployments whose dialect
    the classifier had to work around in the first place.

    Returns False when nothing could be parsed at all. That is a known limit
    rather than a guarantee: ``parse_sql_type`` has a fallback that rescues
    vendor SELECTs sqlglot cannot parse, so a statement it rescues is treated as
    a read while this function cannot see inside it. Detecting that would mean
    refusing unparseable reads outright, which is the very thing the fallback
    exists to avoid.
    """
    for read in (parse_dialect(dialect), None):
        try:
            parsed = sqlglot.parse_one(sql, read=read)
        except Exception:
            continue
        if parsed is None:
            continue
        return any(
            isinstance(node, (expressions.Insert, expressions.Update, expressions.Delete, expressions.Merge))
            for node in parsed.walk()
        )
    return False


def validate_read_only_sql(sql: str, dialect: str) -> tuple[Optional[str], SQLType]:
    """Classify one SQL statement and identify read-only safety violations.

    Returns a violation code plus the detected SQL type. A ``None`` violation
    means the statement is one read-only SELECT, metadata query, or EXPLAIN.
    Callers own their user-facing error wording while sharing the security
    checks themselves.
    """
    cleaned = strip_sql_comments(sql).strip()
    normalized_sql = cleaned.rstrip(";").strip()
    if normalized_sql and not is_single_statement(normalized_sql):
        return READ_ONLY_MULTI_STATEMENT, SQLType.UNKNOWN

    sql_type = parse_sql_type(sql, dialect)
    if sql_type not in (SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN):
        return READ_ONLY_NON_READ, sql_type

    if normalized_sql[:7].upper() == "EXPLAIN":
        # `EXPLAIN ANALYZE <write>` RUNS the write — that is what ANALYZE means
        # in postgres and mysql. Classifying the whole thing as EXPLAIN and
        # stopping there let `EXPLAIN ANALYZE DELETE FROM orders` through every
        # gate built on this helper, on a deployment whose entire promise is
        # that no non-read statement may run.
        #
        # Refused whenever the explained statement is not itself a read, rather
        # than only when ANALYZE is spelled out. Deciding on the option keyword
        # would mean enumerating every dialect's spelling correctly forever, and
        # getting that wrong fails open; a read-only caller has no need to
        # explain a write either way.
        #
        # Keyed on the statement text, not on `sql_type`: sqlglot parses EXPLAIN
        # to a `Describe` on mysql, which classifies as METADATA_SHOW rather
        # than EXPLAIN, so branching on the type would have left mysql — one of
        # the two dialects where ANALYZE actually executes — uncovered.
        inner = _EXPLAIN_PREFIX_RE.sub("", normalized_sql, count=1).strip()
        if not inner:
            return READ_ONLY_NON_READ, sql_type
        inner_type = parse_sql_type(inner, dialect)
        if inner_type not in (SQLType.SELECT, SQLType.METADATA_SHOW, SQLType.EXPLAIN) or _embeds_write_statement(
            inner, dialect
        ):
            # Reuses READ_ONLY_NON_READ rather than adding a code: callers map
            # codes to their own wording with a dict lookup, and an unmapped
            # code would raise KeyError inside a security gate. Returning the
            # inner type also makes the refusal say `delete`, not `explain`.
            return READ_ONLY_NON_READ, inner_type

    if sql_type == SQLType.METADATA_SHOW:
        first_word = cleaned.split()[0].upper() if cleaned else ""
        if first_word == "PRAGMA" and "=" in cleaned:
            return READ_ONLY_WRITABLE_PRAGMA, sql_type

    # Last, because it is the most expensive check and the cheap ones have
    # already rejected the obvious cases: a statement that reads at the root can
    # still write inside a CTE.
    if _embeds_write_statement(normalized_sql, dialect):
        return READ_ONLY_NON_READ, sql_type

    return None, sql_type


# ``SQLType.DDL`` refinements for the permission layer: statements that
# destroy data or schema get their own kind so they can be gated separately
# from benign DDL (COMMENT/GRANT/ANALYZE/...). RENAME maps to ``alter`` — it
# is the same ALTER-family schema mutation under a different keyword.
_DDL_KIND_KEYWORDS: Dict[str, str] = {
    "CREATE": "create",
    "ALTER": "alter",
    "DROP": "drop",
    "TRUNCATE": "truncate",
    "RENAME": "alter",
}


def deployment_read_only_refusal(agent_config: Any, sql: str, dialect: str) -> Optional[str]:
    """Refusal message when ``agent.sql_read_only`` forbids ``sql``, else ``None``.

    For the paths that reach a connector directly instead of going through
    ``DBFuncTool`` — the workflow ``execute_sql`` node and the output tool's
    revised-SQL check. Those bypass every tool-layer gate, so without this the
    deployment-wide switch simply does not apply to them, and the promise in
    ``docs/configuration/sql_policy.md`` that no entry point may run a non-read
    statement is false.

    Lives here, next to the checks themselves, so the next direct-connector
    caller has one obvious thing to call rather than a fourth hand-rolled copy
    of the same rule. ``agent_config`` is read duck-typed: several of these
    callers accept host-supplied or partially built configs.

    A single message rather than one per violation code: these callers answer a
    workflow, not a model that might retry differently, and the specific code is
    logged for the operator instead.
    """
    if not coerce_bool(getattr(agent_config, "sql_read_only", False), False):
        return None
    violation, sql_type = validate_read_only_sql(sql, dialect)
    if not violation:
        return None
    logger.warning(
        "workflow SQL rejected by read-only policy",
        sql_type=sql_type.value,
        source="deployment",
        rule=violation,
    )
    return (
        "This deployment is read-only (agent.sql_read_only): only single "
        f"SELECT/SHOW/DESCRIBE/EXPLAIN statements may run. Detected: {sql_type.value}."
    )


def parse_sql_statement_kind(sql: str, dialect: str = "") -> str:
    """Fine-grained statement kind for the permission layer.

    Same as ``parse_sql_type(...).value`` except two refinements the
    ``execute_sql`` permission gate needs:

    * ``SQLType.DDL`` is split by leading keyword into ``create`` / ``alter``
      / ``drop`` / ``truncate`` (RENAME counts as ``alter``); everything else
      (COMMENT/GRANT/REVOKE/ANALYZE/VACUUM/...) stays ``ddl``.
    * ``REPLACE`` statements (folded into ``SQLType.INSERT`` by
      ``parse_sql_type``) become ``replace`` — REPLACE INTO deletes matched
      rows before re-inserting, so it must not inherit INSERT's class.

    Only the first statement is classified (same contract as
    ``parse_sql_type``); the tool layer's multi-statement rejection is the
    backstop for trailing statements.

    Returns one of: ``select``, ``metadata``, ``explain``, ``insert``,
    ``replace``, ``create``, ``ddl``, ``context_set``, ``update``,
    ``delete``, ``merge``, ``drop``, ``truncate``, ``alter``, ``unknown``.
    """
    sql_type = parse_sql_type(sql, dialect)
    if sql_type not in (SQLType.DDL, SQLType.INSERT, SQLType.CONTENT_SET):
        return sql_type.value

    first_statement = _first_statement(sql.strip())
    match = re.match(r"\s*([A-Za-z_]+)", first_statement)
    keyword = match.group(1).upper() if match else ""

    if sql_type == SQLType.INSERT:
        return "replace" if keyword == "REPLACE" else SQLType.INSERT.value
    if sql_type == SQLType.CONTENT_SET:
        # sqlglot parses dialect-specific statements it cannot model as a
        # generic Command, which ``parse_sql_type`` folds into CONTENT_SET
        # (e.g. MySQL ``RENAME TABLE``). A destructive leading keyword must
        # not ride that fold into the write class.
        return _DDL_KIND_KEYWORDS.get(keyword, SQLType.CONTENT_SET.value)
    return _DDL_KIND_KEYWORDS.get(keyword, "ddl")


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
        if normalized_dialect == "duckdb":
            if database:
                result["database_name"] = database
                result["schema_name"] = identifier
                result["target"] = "schema"
            else:
                result["schema_name"] = identifier
                result["target"] = "schema"
                result["fuzzy"] = True
            return result

        if normalized_dialect == "mysql":
            result["database_name"] = identifier
            result["target"] = "database"
            return result

        if normalized_dialect == "starrocks":
            if catalog or (database and not catalog):
                result["catalog_name"] = catalog or database
                result["database_name"] = identifier
            else:
                result["database_name"] = identifier
            result["target"] = "database"
            return result

        if normalized_dialect == "snowflake":
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
            if normalized_dialect == "duckdb" and not database:
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
