import re
from typing import Any, Dict, List

import sqlglot
from sqlglot import expressions
from sqlglot.expressions import CTE, Table

from datus.utils.constants import DBType, SQLType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def parse_dialect(dialect: str = DBType.SNOWFLAKE) -> str:
    """Parse columns from SQL."""
    dialect = dialect.lower()
    if dialect == DBType.POSTGRESQL:
        dialect = DBType.POSTGRES
    if dialect == DBType.SQLSERVER or dialect == DBType.MSSQL:
        return DBType.MSSQL
    return dialect


def parse_metadata(sql: str, dialect: str = DBType.SNOWFLAKE) -> Dict[str, Any]:
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
    if dialect == DBType.MSSQL:
        return parse_sqlserver_metadata(sql)

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


def parse_sqlserver_metadata(sql: str) -> Dict[str, Any]:
    """Parse SQL Server DDL and return structured table and column information."""
    try:
        result = {"table": {"name": ""}, "columns": []}

        # Clean up the SQL
        sql = sql.strip()

        # Extract table name
        table_pattern = r"CREATE\s+TABLE\s+(?:\[([^\]]+)\]|([^\s(]+))"
        table_match = re.search(table_pattern, sql, re.IGNORECASE)
        if table_match:
            result["table"]["name"] = table_match.group(1) or table_match.group(2)

        # Extract column definitions
        column_pattern = r"\[([^\]]+)\]\s+([^\s,]+)(?:\s+NOT\s+NULL|\s+NULL)?(?:\s*,\s*|\)\s*$)"
        column_matches = re.finditer(column_pattern, sql)

        for match in column_matches:
            col_name = match.group(1)
            col_type = match.group(2)

            col_dict = {"name": col_name, "type": col_type}

            # Look for column comment
            comment_pattern = rf"\[{re.escape(col_name)}\].*?--\s*([^\n]+)"
            comment_match = re.search(comment_pattern, sql)
            if comment_match:
                col_dict["comment"] = comment_match.group(1).strip()

            result["columns"].append(col_dict)

        return result

    except Exception as e:
        logger.error(f"Error parsing SQL Server SQL: {sql};;; {e}")
        return {"table": {"name": ""}, "columns": []}


def extract_table_names(sql, dialect=DBType.SNOWFLAKE) -> List[str]:
    """
    Extract fully qualified table names (database.schema.table) from SQL.
    Returns a list of unique table names with original case preserved.
    Filters out CTE (Common Table Expression) tables.
    """
    dialect = parse_dialect(dialect)
    # Parse the SQL using sqlglot
    parsed = sqlglot.parse_one(sql, dialect=dialect)
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

        if dialect == DBType.SQLITE:
            table_names.append(table_name)
        elif dialect in [DBType.MYSQL, DBType.ORACLE, DBType.POSTGRES, DBType.MSSQL]:
            table_names.append(table_name if not db else f"{db}.{table_name}")
        else:
            table_names.append(f"{db}.{schema}.{table_name}")

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
        return f"{database_name}.{table_name}"
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


def _first_statement(sql: str) -> str:
    """Return the first non-empty statement (before the first ';'), with comments removed."""
    s = strip_sql_comments(sql).strip()
    if not s:
        return ""
    return s.split(";", 1)[0].strip()


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
    parsed_expression = sqlglot.parse_one(first_statement, dialect=dialect_name, error_level=sqlglot.ErrorLevel.IGNORE)
    if parsed_expression is None:
        if dialect_name == DBType.STARROCKS.value and _metadata_pattern().match(first_statement):
            return SQLType.METADATA_SHOW
        # Return UNKNOWN instead of raising exception for CLI usage
        return SQLType.UNKNOWN

    if isinstance(parsed_expression, expressions.Select):
        return SQLType.SELECT
    elif isinstance(parsed_expression, expressions.Values):
        return SQLType.SELECT
    elif isinstance(parsed_expression, expressions.Insert):
        return SQLType.INSERT
    elif isinstance(parsed_expression, expressions.Merge):
        return SQLType.MERGE
    elif isinstance(parsed_expression, expressions.Update):
        return SQLType.UPDATE
    elif isinstance(parsed_expression, expressions.Delete):
        return SQLType.DELETE
    elif isinstance(
        parsed_expression,
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
    elif isinstance(parsed_expression, (expressions.Describe, expressions.Show, expressions.Pragma)):
        return SQLType.METADATA_SHOW
    elif isinstance(parsed_expression, expressions.Command):
        command_name = str(parsed_expression.args.get("this") or "").upper()
        if command_name in {"SHOW", "DESC", "DESCRIBE"}:
            return SQLType.METADATA_SHOW
        if command_name == "EXPLAIN":
            return SQLType.EXPLAIN
        if command_name == "REPLACE":
            return SQLType.INSERT
        if command_name in {"CALL", "EXEC", "EXECUTE"}:
            return SQLType.CONTENT_SET
        return SQLType.CONTENT_SET
    elif isinstance(
        parsed_expression, (expressions.Use, expressions.Transaction, expressions.Commit, expressions.Rollback)
    ):
        return SQLType.CONTENT_SET
    else:
        return SQLType.UNKNOWN
