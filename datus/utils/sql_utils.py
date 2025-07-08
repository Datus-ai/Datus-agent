import re
from typing import Any, Dict, List

import sqlglot
from sqlglot.expressions import CTE, Table

from .loggings import get_logger

logger = get_logger(__name__)


def parse_dialect(dialect: str = "snowflake") -> str:
    """Parse columns from SQL."""
    dialect = dialect.lower()
    if dialect == "postgresql":
        dialect = "postgres"
    if dialect == "sqlserver" or dialect == "mssql":
        return "mssql"
    return dialect


def parse_metadata(sql: str, dialect: str = "snowflake") -> Dict[str, Any]:
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
    if dialect == "mssql":
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


def extract_table_names(sql, dialect="snowflake") -> List[str]:
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

        if dialect == "sqlite":
            table_names.append(table_name)
        elif dialect in ["mysql", "oracle", "postgres", "mssql"]:
            table_names.append(table_name if not db else f"{db}.{table_name}")
        else:
            table_names.append(f"{db}.{schema}.{table_name}")

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
    if dialect == "sqlite":
        return f"{database_name}.{table_name}"
    elif dialect == "duckdb":
        return f"{database_name}.{schema_name}.{table_name}"
    elif dialect == "mysql":
        return f"{catalog_name}.{database_name}.{table_name}" if catalog_name else f"{database_name}.{table_name}"
    elif dialect == "oracle" or dialect.startswith("postgre"):
        return f"{database_name}.{schema_name}.{table_name}"
    elif dialect == "snowflake":
        return (
            f"{catalog_name}.{database_name}.{schema_name}.{table_name}"
            if catalog_name
            else f"{database_name}.{schema_name}.{table_name}"
        )
    elif dialect == "databricks":
        return f"{catalog_name}.{schema_name}.{table_name}" if catalog_name else f"{schema_name}.{table_name}"
    return table_name
