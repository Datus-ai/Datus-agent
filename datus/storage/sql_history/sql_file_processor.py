# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import glob
import os
import re
from typing import Any, Dict, List, Tuple

import sqlglot

from datus.utils.constants import SQLType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_sql_type

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
    Split by semicolons first, then extract optional comments for each SQL.

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

    # First, split by semicolons to get SQL statements
    sql_blocks = content.split(";")

    pairs = []
    current_line = 1

    for block in sql_blocks:
        block = block.strip()
        if not block:
            continue

        # Split block into lines to extract comments and SQL
        lines = block.split("\n")
        comment_lines = []
        sql_lines = []
        block_start_line = current_line

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("--"):
                # Comment line - remove all leading dashes
                comment_text = re.sub(r"^-+\s*", "", stripped)
                comment_lines.append(comment_text)
            elif stripped:
                # SQL line
                sql_lines.append(line)

        # Update line counter
        current_line += len(lines)

        # Build comment and SQL
        comment = " ".join(comment_lines).strip() if comment_lines else ""
        sql = "\n".join(sql_lines).strip()

        # Clean up SQL
        sql = re.sub(r"\n\s*\n", "\n", sql)
        sql = sql.strip()

        # Add to pairs if SQL is not empty
        if sql:
            pairs.append((comment, sql, block_start_line))

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

                # Check SQL type - only process SELECT queries
                try:
                    sql_type = parse_sql_type(sql_cleaned_parens, "mysql")
                    if sql_type != SQLType.SELECT:
                        logger.debug(f"Skipping non-SELECT SQL (type: {sql_type}) at {sql_file}:{line_num}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to parse SQL type at {sql_file}:{line_num}: {str(e)}")
                    continue

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
