import glob
import os
import re
from typing import Any, Dict, List, Tuple

import sqlglot

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def remove_outer_parentheses(sql: str) -> str:
    """
    Remove unmatched parentheses from the beginning and end of SQL.

    Args:
        sql: SQL statement string

    Returns:
        SQL statement with unmatched parentheses removed
    """
    if not sql or not sql.strip():
        return sql

    original_sql = sql
    sql = sql.strip()

    # Count total parentheses to identify unmatched ones
    open_count = sql.count("(")
    close_count = sql.count(")")

    # Remove trailing unmatched closing parentheses
    while close_count > open_count and sql.endswith(")"):
        sql = sql[:-1].rstrip()
        close_count -= 1
        logger.debug(f"Removed trailing ')': {sql[-50:] if len(sql) > 50 else sql}")

    # Remove leading unmatched opening parentheses
    while open_count > close_count and sql.startswith("("):
        sql = sql[1:].lstrip()
        open_count -= 1
        logger.debug(f"Removed leading '(': {sql[:50]}...")

    # Only log if we made changes
    if sql != original_sql.strip():
        logger.debug("Fixed unmatched parentheses in SQL")

    return sql


def parse_comment_sql_pairs(file_path: str) -> List[Tuple[str, str, int]]:
    """
    Parse a SQL file to extract comment-SQL pairs with line numbers.

    Args:
        file_path: Path to the SQL file

    Returns:
        List of tuples (comment, sql, line_number)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, "r", encoding="gbk") as f:
            content = f.read()

    # Split content by comment lines starting with --
    # Keep track of line numbers for each section
    lines = content.split("\n")
    pairs = []

    i = 0
    while i < len(lines):
        # Skip empty lines
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines):
            break

        # Check if this line starts a comment block
        if lines[i].strip().startswith("--"):
            comment_start_line = i + 1  # 1-indexed line number
            comment_lines = []

            # Collect all consecutive comment lines
            while i < len(lines) and lines[i].strip().startswith("--"):
                comment_lines.append(lines[i].strip().removeprefix("--").strip())
                i += 1

            # Collect SQL lines until next comment or end of file
            sql_lines = []

            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("--"):
                    # Next comment block found, stop collecting SQL
                    break
                if line:  # Only add non-empty lines
                    sql_lines.append(lines[i])
                i += 1

            # Process the collected comment and SQL
            if comment_lines:
                comment = " ".join(comment_lines).strip()
                sql = "\n".join(sql_lines).strip()

                # Remove multiple empty lines and clean up SQL
                sql = re.sub(r"\n\s*\n", "\n", sql)
                sql = sql.strip()

                if comment and sql:
                    pairs.append((comment, sql, comment_start_line))
        else:
            # Skip non-comment lines at the beginning
            i += 1

    return pairs


def preprocess_parameterized_sql(sql: str) -> str:
    """
    Preprocess SQL to replace parameter placeholders with dummy values for validation.

    Args:
        sql: SQL string that may contain parameter placeholders

    Returns:
        SQL string with parameters replaced by dummy values
    """
    import re

    # Replace #parameter# style parameters with dummy values
    sql = re.sub(r"#\w+#", "'2023-01-01'", sql)

    # Replace other common parameter styles - but be more careful to avoid time formats
    # Only replace :param if it's not part of a time format (like 04:00:00)
    sql = re.sub(r"(?<!\d):\w+(?!\d)", "'dummy_value'", sql)  # :param (not preceded/followed by digits)
    sql = re.sub(r"@\w+\b", "'dummy_value'", sql)  # @param (word boundary)
    sql = re.sub(r"\$\{\w+\}", "'dummy_value'", sql)  # ${param}

    return sql


def validate_sql(sql: str) -> Tuple[bool, str, str]:
    """
    Validate SQL using MySQL, Hive, and Spark dialects.

    Args:
        sql: Raw SQL string

    Returns:
        Tuple of (is_valid, cleaned_sql, error_message)
    """
    # Preprocess SQL to handle parameter placeholders
    preprocessed_sql = preprocess_parameterized_sql(sql)

    # Try MySQL, Hive, Spark dialects with sqlglot
    dialects_to_try = ["mysql", "hive", "spark"]

    sqlglot_errors = []

    for dialect in dialects_to_try:
        try:
            parsed = sqlglot.parse(preprocessed_sql, read=dialect)
            if not parsed or not parsed[0]:
                continue

            # Check if we have valid parsed statements
            valid_statements = []
            for stmt in parsed:
                if stmt:
                    valid_statements.append(stmt)

            if not valid_statements:
                continue

            # Transpile back to get cleaned SQL (use original SQL to preserve parameters)
            cleaned_sql = sqlglot.transpile(sql, read=dialect, pretty=True)[0]
            return True, cleaned_sql, ""

        except Exception as e:
            # Strip ANSI color codes from error messages
            error_msg = str(e)
            error_msg = re.sub(r"\x1b\[[0-9;]*m", "", error_msg)
            sqlglot_errors.append(f"\n\t{dialect}: {error_msg}")

    # All dialects failed
    return False, "", f"SQL validation errors: {'; '.join(sqlglot_errors)}"


def process_sql_files(sql_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process all SQL files in a directory with strict validation.

    Args:
        sql_dir: Directory containing SQL files

    Returns:
        Tuple of (valid_entries, invalid_entries)
    """
    if not os.path.exists(sql_dir):
        raise ValueError(f"SQL directory not found: {sql_dir}")

    sql_files = glob.glob(os.path.join(sql_dir, "*.sql"))
    if not sql_files:
        raise ValueError(f"No SQL files found in directory: {sql_dir}")

    logger.info(f"Found {len(sql_files)} SQL files to process")

    valid_entries = []
    invalid_entries = []

    for sql_file in sql_files:
        logger.info(f"Processing file: {sql_file}")

        try:
            pairs = parse_comment_sql_pairs(sql_file)
            logger.info(f"Extracted {len(pairs)} comment-SQL pairs from {os.path.basename(sql_file)}")

            for comment, sql, line_num in pairs:
                # Remove outer parentheses before validation
                sql_cleaned_parens = remove_outer_parentheses(sql)
                is_valid, cleaned_sql, error_msg = validate_sql(sql_cleaned_parens)

                if is_valid:
                    valid_entries.append(
                        {
                            "comment": comment or "",
                            "sql": cleaned_sql,
                            "filepath": sql_file,
                        }
                    )
                else:
                    invalid_entries.append(
                        {
                            "comment": comment,
                            "sql": sql_cleaned_parens,  # Use the processed SQL
                            "filepath": sql_file,
                            "error": error_msg,
                            "line_number": line_num,
                        }
                    )

        except Exception as e:
            logger.error(f"Error processing file {sql_file}: {str(e)}")
            invalid_entries.append(
                {
                    "comment": "",
                    "sql": "",
                    "filepath": sql_file,
                    "error": f"File processing error: {str(e)}",
                    "line_number": 1,
                }
            )

    # Log summary
    logger.info(f"Processing complete: {len(valid_entries)} valid, {len(invalid_entries)} invalid SQL entries")

    # Log invalid entries for review
    if invalid_entries:
        log_invalid_entries(invalid_entries)

    return valid_entries, invalid_entries


def log_invalid_entries(invalid_entries: List[Dict[str, Any]]):
    """Log invalid SQL entries to a separate log file for manual review."""
    log_file = "sql_processing_errors.log"

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"SQL Processing Errors - {len(invalid_entries)} invalid entries\n")
            f.write("=" * 80 + "\n\n")

            for i, entry in enumerate(invalid_entries, 1):
                f.write(f"[{i}] Invalid SQL Entry\n")
                f.write(f"File: {entry['filepath']}\n")
                line_info = f" (line {entry.get('line_number', 'unknown')})" if "line_number" in entry else ""
                f.write(f"Comment: {entry['comment']}{line_info}\n")
                f.write(f"Error: {entry['error']}\n")
                f.write(f"SQL:\n{entry['sql']}\n")
                f.write("-" * 80 + "\n\n")

        logger.warning(f"Invalid SQL entries logged to: {log_file}")

    except Exception as e:
        logger.error(f"Failed to write invalid SQL log: {str(e)}")
