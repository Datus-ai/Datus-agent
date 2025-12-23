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


def parse_comment_sql_pairs(file_path: str) -> List[Tuple[str, str, int]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, "r", encoding="gbk") as f:
            content = f.read()

    # Split into lines first to handle comments properly
    lines = content.split("\n")

    # Build blocks by tracking comments and SQL, splitting only on non-comment semicolons
    blocks = []
    current_block_lines = []
    block_start_line = 1

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Check if this line contains a statement-ending semicolon (not in comment)
        has_semicolon = False
        if ";" in line:
            # Find first semicolon and comment position
            comment_pos = line.find("--")
            semicolon_pos = line.find(";")

            # Semicolon is valid statement terminator if:
            # 1. No comment on this line, OR
            # 2. Semicolon appears before the comment
            if comment_pos == -1 or semicolon_pos < comment_pos:
                has_semicolon = True

        current_block_lines.append(line)

        # If we found a statement-ending semicolon, finalize this block
        if has_semicolon:
            block_content = "\n".join(current_block_lines)
            blocks.append((block_content, block_start_line))
            current_block_lines = []
            block_start_line = line_num + 1

    # Handle any remaining lines as a final block
    if current_block_lines:
        block_content = "\n".join(current_block_lines)
        if block_content.strip():
            blocks.append((block_content, block_start_line))

    # Now process each block to extract comments and SQL
    pairs = []

    for block_content, block_line_num in blocks:
        comment_lines = []
        sql_lines = []

        for line in block_content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("--"):
                # Comment line - remove all leading dashes
                comment_text = re.sub(r"^-+\s*", "", stripped)
                comment_lines.append(comment_text)
            elif stripped:
                # SQL line - keep as-is (don't remove semicolons per-line to preserve string literals)
                sql_lines.append(line)

        # Build comment and SQL
        comment = " ".join(comment_lines).strip() if comment_lines else ""
        sql = "\n".join(sql_lines)

        # Clean up SQL: remove trailing whitespace and statement-terminating semicolon
        sql = sql.rstrip()  # Remove trailing whitespace
        if sql.endswith(";"):
            sql = sql[:-1].rstrip()  # Remove trailing semicolon and any whitespace before it

        # Remove excessive blank lines
        sql = re.sub(r"\n\s*\n", "\n", sql)
        sql = sql.strip()

        # Add to pairs if SQL is not empty
        if sql:
            pairs.append((comment, sql, block_line_num))

    return pairs


def preprocess_parameterized_sql(sql: str) -> str:
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
                # Check SQL type - only process SELECT queries
                try:
                    sql_type = parse_sql_type(sql, "mysql")
                    if sql_type != SQLType.SELECT:
                        logger.debug(f"Skipping non-SELECT SQL (type: {sql_type}) at {sql_file}:{line_num}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to parse SQL type at {sql_file}:{line_num}: {str(e)}")
                    continue

                is_valid, cleaned_sql, error_msg = validate_sql(sql)

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
                            "sql": sql,
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
