# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sqlglot

from datus.utils.constants import SQLType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_sql_type

logger = get_logger(__name__)


def _find_effective_semicolon(line: str, in_block_comment: bool) -> Tuple[int, bool]:
    """
    Find the position of an effective semicolon (not inside any comment or string literal) in a line.

    Args:
        line: The line to check
        in_block_comment: Whether we're currently inside a block comment from previous lines

    Returns:
        Tuple of (semicolon_position or -1 if none, updated in_block_comment state)
    """
    i = 0
    in_single_quote = False
    in_double_quote = False

    while i < len(line):
        if in_block_comment:
            # Look for end of block comment
            end_pos = line.find("*/", i)
            if end_pos == -1:
                # Still in block comment, no effective semicolon possible
                return -1, True
            # Exit block comment and continue scanning
            in_block_comment = False
            i = end_pos + 2
        elif in_single_quote:
            # Inside single-quoted string - look for closing quote
            if line[i] == "'":
                # Check for escaped single quote ('')
                if i + 1 < len(line) and line[i + 1] == "'":
                    # Escaped quote, skip both
                    i += 2
                    continue
                # Closing quote found
                in_single_quote = False
            i += 1
        elif in_double_quote:
            # Inside double-quoted identifier - look for closing quote
            if line[i] == '"':
                # Check for escaped double quote ("")
                if i + 1 < len(line) and line[i + 1] == '"':
                    # Escaped quote, skip both
                    i += 2
                    continue
                # Closing quote found
                in_double_quote = False
            i += 1
        else:
            # Not in comment or quote - check for quote starts, comment starts, or semicolon
            if line[i] == "'":
                in_single_quote = True
                i += 1
            elif line[i] == '"':
                in_double_quote = True
                i += 1
            elif line[i : i + 2] == "--":
                # Rest of line is comment, no more effective semicolons
                return -1, False
            elif line[i : i + 2] == "/*":
                in_block_comment = True
                i += 2
            elif line[i] == ";":
                # Found effective semicolon, but need to continue to update block comment state
                semicolon_pos = i
                # Continue scanning to update in_block_comment state for next line
                i += 1
                in_single_quote = False
                in_double_quote = False
                while i < len(line):
                    if in_block_comment:
                        end_pos = line.find("*/", i)
                        if end_pos == -1:
                            return semicolon_pos, True
                        in_block_comment = False
                        i = end_pos + 2
                    elif in_single_quote:
                        if line[i] == "'":
                            if i + 1 < len(line) and line[i + 1] == "'":
                                i += 2
                                continue
                            in_single_quote = False
                        i += 1
                    elif in_double_quote:
                        if line[i] == '"':
                            if i + 1 < len(line) and line[i + 1] == '"':
                                i += 2
                                continue
                            in_double_quote = False
                        i += 1
                    else:
                        if line[i] == "'":
                            in_single_quote = True
                            i += 1
                        elif line[i] == '"':
                            in_double_quote = True
                            i += 1
                        elif line[i : i + 2] == "--":
                            return semicolon_pos, False
                        elif line[i : i + 2] == "/*":
                            in_block_comment = True
                            i += 2
                        else:
                            i += 1
                return semicolon_pos, in_block_comment
            else:
                i += 1
    return -1, in_block_comment


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
    in_block_comment = False

    for line_num, line in enumerate(lines, 1):
        # Check if this line contains a statement-ending semicolon (not in comment)
        semicolon_pos, in_block_comment = _find_effective_semicolon(line, in_block_comment)
        has_semicolon = semicolon_pos >= 0

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

    # Process each block - keep entire block as SQL (including comments)
    pairs = []

    for block_content, block_line_num in blocks:
        # Keep the entire block as SQL (including inline comments)
        sql = block_content

        # Clean up SQL: remove trailing whitespace and statement-terminating semicolon
        sql = sql.rstrip()
        if sql.endswith(";"):
            sql = sql[:-1].rstrip()

        # Remove excessive blank lines
        sql = re.sub(r"\n\s*\n", "\n", sql)
        sql = sql.strip()

        # Add to pairs if SQL is not empty (comment is empty since it's embedded in SQL)
        if sql:
            pairs.append(("", sql, block_line_num))

    return pairs


def preprocess_parameterized_sql(sql: str) -> str:
    import re

    # First, protect string literals by replacing them with placeholders
    # This prevents replacing patterns inside quoted strings (like time formats)
    string_literals = []
    placeholder_template = "__STRING_LITERAL_{}_PLACEHOLDER__"

    def save_string(match):
        idx = len(string_literals)
        string_literals.append(match.group(0))
        return placeholder_template.format(idx)

    # Match single-quoted and double-quoted strings (handling SQL-style escaped quotes)
    # SQL uses doubled quotes for escaping: 'it''s' and "say ""hello"""
    sql = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", save_string, sql)

    # Replace #parameter# style parameters with dummy values
    sql = re.sub(r"#\w+#", "'2023-01-01'", sql)

    # Replace other common parameter styles
    sql = re.sub(r":\w+", "'dummy_value'", sql)  # :param
    sql = re.sub(r"@\w+\b", "'dummy_value'", sql)  # @param (word boundary)
    sql = re.sub(r"\$\{\w+\}", "'dummy_value'", sql)  # ${param}

    # Restore string literals
    for idx, literal in enumerate(string_literals):
        sql = sql.replace(placeholder_template.format(idx), literal)

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


def process_sql_items(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid_entries: List[Dict[str, Any]] = []
    invalid_entries: List[Dict[str, Any]] = []

    for item in items:
        sql = (item.get("sql") or "").strip()
        if not sql:
            continue

        comment = item.get("comment") or ""
        filepath = item.get("filepath") or ""
        line_number = item.get("line_number", 1)

        # Check SQL type - only process SELECT queries
        try:
            sql_type = parse_sql_type(sql, "mysql")
            if sql_type != SQLType.SELECT:
                logger.debug(f"Skipping non-SELECT SQL (type: {sql_type}) at {filepath}:{line_number}")
                continue
        except Exception as e:
            logger.warning(f"Failed to parse SQL type at {filepath}:{line_number}: {str(e)}")
            continue

        is_valid, cleaned_sql, error_msg = validate_sql(sql)

        if is_valid:
            cleaned_item = dict(item)
            cleaned_item["comment"] = comment
            cleaned_item["sql"] = cleaned_sql
            cleaned_item["filepath"] = filepath
            cleaned_item.pop("line_number", None)
            cleaned_item.pop("error", None)
            valid_entries.append(cleaned_item)
        else:
            invalid_item = dict(item)
            invalid_item["comment"] = comment
            invalid_item["sql"] = sql
            invalid_item["filepath"] = filepath
            invalid_item["error"] = error_msg
            invalid_item["line_number"] = line_number
            invalid_entries.append(invalid_item)

    return valid_entries, invalid_entries


def process_sql_files(sql_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not os.path.exists(sql_dir):
        raise ValueError(f"SQL directory not found: {sql_dir}")
    sql_dir_path = Path(sql_dir).expanduser().resolve()
    if sql_dir_path.is_dir():
        sql_files = glob.glob(os.path.join(sql_dir, "*.sql"))
    elif sql_dir_path.is_file() and sql_dir_path.suffix.lower() == ".sql":
        sql_files = [sql_dir]
    else:
        sql_files = []
    if not sql_files:
        raise ValueError(f"No SQL files found in directory: {sql_dir}")

    logger.info(f"Found {len(sql_files)} SQL files to process")

    valid_entries: List[Dict[str, Any]] = []
    invalid_entries: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []

    for sql_file in sql_files:
        logger.info(f"Processing file: {sql_file}")

        try:
            pairs = parse_comment_sql_pairs(sql_file)
            logger.info(f"Extracted {len(pairs)} SQL blocks from {os.path.basename(sql_file)}")

            for comment, sql, line_num in pairs:
                items.append(
                    {
                        "comment": comment or "",
                        "sql": sql,
                        "filepath": sql_file,
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

    processed_valid, processed_invalid = process_sql_items(items)
    valid_entries.extend(processed_valid)
    invalid_entries.extend(processed_invalid)

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
