# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jinja2
import jinja2.meta

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def extract_template_parameters(template_content: str) -> List[Dict[str, str]]:
    """Extract Jinja2 variable names from a template using AST parsing.

    Args:
        template_content: Raw Jinja2 template content

    Returns:
        List of parameter definitions, e.g. [{"name": "start_date"}, {"name": "end_date"}]
    """
    env = jinja2.Environment()
    try:
        ast = env.parse(template_content)
        variables = jinja2.meta.find_undeclared_variables(ast)
        return [{"name": var} for var in sorted(variables)]
    except jinja2.TemplateSyntaxError:
        # If parsing fails, fall back to regex extraction of {{ var }} patterns
        pattern = r"\{\{\s*(\w+)\s*\}\}"
        matches = set(re.findall(pattern, template_content))
        return [{"name": var} for var in sorted(matches)]


def validate_template(template_content: str) -> Tuple[bool, str]:
    """Validate Jinja2 template syntax.

    Args:
        template_content: Raw Jinja2 template content

    Returns:
        Tuple of (is_valid, error_message)
    """
    env = jinja2.Environment()
    try:
        env.parse(template_content)
        return True, ""
    except jinja2.TemplateSyntaxError as e:
        return False, f"Jinja2 syntax error: {e}"


def parse_template_blocks(file_path: str) -> List[Tuple[str, str, int]]:
    """Parse a template file into template blocks separated by semicolons.

    Supports both single-template files and multi-template files where
    templates are separated by ';' (similar to SQL file processing).

    Args:
        file_path: Path to the .j2/.jinja2 file

    Returns:
        List of (comment, template_content, line_number) tuples
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk") as f:
            content = f.read()

    # Split by semicolons, but respect Jinja2 blocks and strings
    blocks = _split_by_semicolons(content)

    pairs = []
    for block_content, block_line_num in blocks:
        template = block_content.strip()
        if template:
            pairs.append(("", template, block_line_num))

    return pairs


def _split_by_semicolons(content: str) -> List[Tuple[str, int]]:
    """Split content by effective semicolons (not inside Jinja2 blocks or strings).

    Args:
        content: File content

    Returns:
        List of (block_content, start_line_number) tuples
    """
    lines = content.split("\n")
    blocks = []
    current_block_lines = []
    block_start_line = 1
    in_jinja_block = 0  # Nesting depth for {% ... %}
    in_jinja_comment = False  # Inside {# ... #}

    for line_num, line in enumerate(lines, 1):
        semicolon_pos = _find_effective_semicolon_j2(line, in_jinja_block, in_jinja_comment)

        # Update Jinja2 block nesting state
        in_jinja_block, in_jinja_comment = _update_jinja_state(line, in_jinja_block, in_jinja_comment)

        current_block_lines.append(line)

        if semicolon_pos >= 0 and in_jinja_block == 0 and not in_jinja_comment:
            block_content = "\n".join(current_block_lines)
            # Remove the trailing semicolon
            block_content = block_content.rstrip()
            if block_content.endswith(";"):
                block_content = block_content[:-1].rstrip()
            blocks.append((block_content, block_start_line))
            current_block_lines = []
            block_start_line = line_num + 1

    # Handle remaining content as the last block
    if current_block_lines:
        block_content = "\n".join(current_block_lines)
        block_content = block_content.rstrip()
        if block_content.endswith(";"):
            block_content = block_content[:-1].rstrip()
        if block_content.strip():
            blocks.append((block_content, block_start_line))

    return blocks


def _find_effective_semicolon_j2(line: str, in_jinja_block: int, in_jinja_comment: bool) -> int:
    """Find position of an effective semicolon in a line, respecting Jinja2 syntax.

    Args:
        line: The line to check
        in_jinja_block: Current Jinja2 block nesting depth
        in_jinja_comment: Whether inside a Jinja2 comment

    Returns:
        Position of semicolon or -1 if none found
    """
    if in_jinja_comment:
        return -1
    if in_jinja_block > 0:
        return -1

    i = 0
    in_single_quote = False
    in_double_quote = False

    while i < len(line):
        if in_single_quote:
            if line[i] == "'" and (i + 1 >= len(line) or line[i + 1] != "'"):
                in_single_quote = False
            i += 1
        elif in_double_quote:
            if line[i] == '"' and (i + 1 >= len(line) or line[i + 1] != '"'):
                in_double_quote = False
            i += 1
        elif line[i] == "'":
            in_single_quote = True
            i += 1
        elif line[i] == '"':
            in_double_quote = True
            i += 1
        elif line[i : i + 2] == "--":
            # SQL line comment, rest of line is comment
            return -1
        elif line[i : i + 2] == "{%":
            # Jinja2 block tag - skip to closing %}
            end_pos = line.find("%}", i + 2)
            if end_pos >= 0:
                i = end_pos + 2
            else:
                return -1  # Unclosed block tag, skip rest
        elif line[i : i + 2] == "{{":
            # Jinja2 expression - skip to closing }}
            end_pos = line.find("}}", i + 2)
            if end_pos >= 0:
                i = end_pos + 2
            else:
                return -1  # Unclosed expression, skip rest
        elif line[i : i + 2] == "{#":
            # Jinja2 comment - skip to closing #}
            end_pos = line.find("#}", i + 2)
            if end_pos >= 0:
                i = end_pos + 2
            else:
                return -1  # Unclosed comment, skip rest
        elif line[i] == ";":
            return i
        else:
            i += 1

    return -1


def _update_jinja_state(line: str, in_jinja_block: int, in_jinja_comment: bool) -> Tuple[int, bool]:
    """Update Jinja2 block/comment state after processing a line.

    Args:
        line: The line to scan
        in_jinja_block: Current block nesting depth
        in_jinja_comment: Whether inside a Jinja2 comment

    Returns:
        Tuple of (updated_block_depth, updated_in_comment)
    """
    i = 0
    while i < len(line):
        if in_jinja_comment:
            end_pos = line.find("#}", i)
            if end_pos >= 0:
                in_jinja_comment = False
                i = end_pos + 2
            else:
                return in_jinja_block, True
        elif line[i : i + 2] == "{#":
            in_jinja_comment = True
            i += 2
        elif line[i : i + 2] == "{%":
            # Check for block-opening tags (for, if, block, macro, etc.)
            end_pos = line.find("%}", i)
            if end_pos >= 0:
                tag_content = line[i + 2 : end_pos].strip()
                if _is_block_opening_tag(tag_content):
                    in_jinja_block += 1
                elif _is_block_closing_tag(tag_content):
                    in_jinja_block = max(0, in_jinja_block - 1)
                i = end_pos + 2
            else:
                i += 2
        else:
            i += 1

    return in_jinja_block, in_jinja_comment


def _is_block_opening_tag(tag_content: str) -> bool:
    """Check if a Jinja2 tag opens a block."""
    opening_keywords = {"for", "if", "block", "macro", "call", "filter", "set"}
    first_word = tag_content.split()[0] if tag_content.split() else ""
    # 'set' only opens a block if it doesn't have '=' (assignment form)
    if first_word == "set" and "=" in tag_content:
        return False
    return first_word in opening_keywords


def _is_block_closing_tag(tag_content: str) -> bool:
    """Check if a Jinja2 tag closes a block."""
    closing_keywords = {"endfor", "endif", "endblock", "endmacro", "endcall", "endfilter", "endset"}
    first_word = tag_content.split()[0] if tag_content.split() else ""
    return first_word in closing_keywords


def process_template_items(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate template items and extract parameters.

    Args:
        items: List of raw template items with 'template', 'filepath', 'comment', 'line_number'

    Returns:
        Tuple of (valid_items, invalid_items)
    """
    valid_entries: List[Dict[str, Any]] = []
    invalid_entries: List[Dict[str, Any]] = []

    for item in items:
        template = (item.get("template") or "").strip()
        if not template:
            continue

        comment = item.get("comment") or ""
        filepath = item.get("filepath") or ""
        line_number = item.get("line_number", 1)

        # Validate Jinja2 syntax
        is_valid, error_msg = validate_template(template)

        if is_valid:
            # Extract parameters
            parameters = extract_template_parameters(template)
            cleaned_item = dict(item)
            cleaned_item["comment"] = comment
            cleaned_item["template"] = template
            cleaned_item["parameters"] = json.dumps(parameters)
            cleaned_item["filepath"] = filepath
            cleaned_item.pop("line_number", None)
            cleaned_item.pop("error", None)
            valid_entries.append(cleaned_item)
        else:
            invalid_item = dict(item)
            invalid_item["comment"] = comment
            invalid_item["template"] = template
            invalid_item["filepath"] = filepath
            invalid_item["error"] = error_msg
            invalid_item["line_number"] = line_number
            invalid_entries.append(invalid_item)

    return valid_entries, invalid_entries


def process_template_files(template_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process all template files in a directory.

    Supports both directory of .j2/.jinja2 files and single file.
    Each file can contain one or more templates separated by ';'.

    Args:
        template_dir: Path to the template files directory or a single file

    Returns:
        Tuple of (valid_items, invalid_items)
    """
    if not os.path.exists(template_dir):
        raise ValueError(f"Template directory not found: {template_dir}")

    template_dir_path = Path(template_dir).expanduser().resolve()
    if template_dir_path.is_dir():
        template_files = glob.glob(os.path.join(template_dir, "*.j2")) + glob.glob(
            os.path.join(template_dir, "*.jinja2")
        )
    elif template_dir_path.is_file() and template_dir_path.suffix.lower() in (".j2", ".jinja2"):
        template_files = [template_dir]
    else:
        template_files = []

    if not template_files:
        raise ValueError(f"No template files (*.j2, *.jinja2) found in directory: {template_dir}")

    logger.info(f"Found {len(template_files)} template files to process")

    valid_entries: List[Dict[str, Any]] = []
    invalid_entries: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []

    for template_file in template_files:
        logger.info(f"Processing file: {template_file}")

        try:
            blocks = parse_template_blocks(template_file)
            logger.info(f"Extracted {len(blocks)} template blocks from {os.path.basename(template_file)}")

            for comment, template, line_num in blocks:
                items.append(
                    {
                        "comment": comment or "",
                        "template": template,
                        "filepath": template_file,
                        "line_number": line_num,
                    }
                )

        except Exception as e:
            logger.error(f"Error processing file {template_file}: {str(e)}")
            invalid_entries.append(
                {
                    "comment": "",
                    "template": "",
                    "filepath": template_file,
                    "error": f"File processing error: {str(e)}",
                    "line_number": 1,
                }
            )

    processed_valid, processed_invalid = process_template_items(items)
    valid_entries.extend(processed_valid)
    invalid_entries.extend(processed_invalid)

    logger.info(f"Processing complete: {len(valid_entries)} valid, {len(invalid_entries)} invalid template entries")

    if invalid_entries:
        log_invalid_entries(invalid_entries)

    return valid_entries, invalid_entries


def log_invalid_entries(invalid_entries: List[Dict[str, Any]]):
    log_file = "template_processing_errors.log"

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Template Processing Errors - {len(invalid_entries)} invalid entries\n")
            f.write("=" * 80 + "\n\n")

            for i, entry in enumerate(invalid_entries, 1):
                f.write(f"[{i}] Invalid Template Entry\n")
                f.write(f"File: {entry['filepath']}\n")
                line_info = f" (line {entry.get('line_number', 'unknown')})" if "line_number" in entry else ""
                f.write(f"Comment: {entry['comment']}{line_info}\n")
                f.write(f"Error: {entry['error']}\n")
                f.write(f"Template:\n{entry['template']}\n")
                f.write("-" * 80 + "\n\n")

        logger.warning(f"Invalid template entries logged to: {log_file}")

    except Exception as e:
        logger.error(f"Failed to write invalid template log: {str(e)}")
