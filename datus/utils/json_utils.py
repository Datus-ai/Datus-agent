# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import json_repair
import pandas as pd

try:  # Optional dependency; available in runtime environments that use Pydantic models.
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - Pydantic is an optional dependency.
    BaseModel = None  # type: ignore[assignment]

try:  # NumPy may not always be installed, but handle it when present.
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is optional.
    np = None  # type: ignore[assignment]


def json2csv(result: Any, columns: Optional[List[str]] = None) -> str:
    """
    Convert JSON data to CSV format.

    Args:
        result: JSON data to convert
        columns: Optional list of columns to include in the CSV

    Returns:
        str: CSV formatted string
    """
    if not result:
        return ""
    if isinstance(result, str):
        if result.strip().startswith("[") or result.strip().startswith("{"):
            result = json_repair.loads(result)
        else:
            return result
    if isinstance(result, dict):
        result = [result]
    if isinstance(result, list):
        with io.StringIO() as output:
            df = pd.DataFrame(result)
            df.to_csv(output, index=False, columns=columns)
            return output.getvalue()
    else:
        raise ValueError(f"Invalid result type: {type(result)}")


def find_matching_bracket(text: str, start_idx: int, open_char: str = "[", close_char: str = "]") -> int:
    """
    Find the matching closing bracket for the opening bracket at start_idx.

    Args:
        text: The text to search in
        start_idx: The index of the opening bracket
        open_char: The opening bracket character
        close_char: The closing bracket character

    Returns:
        int: The index of the matching closing bracket, or -1 if not found
    """
    stack = []
    for i in range(start_idx, len(text)):
        if text[i] == open_char:
            stack.append(open_char)
        elif text[i] == close_char:
            if not stack:
                return -1
            stack.pop()
            if not stack:
                return i
    return -1


def extract_json_object(text: str) -> str:
    """
    Extract the first valid JSON object from the text.

    Args:
        text: The text to extract from

    Returns:
        str: The extracted JSON object string, or empty string if not found
    """
    start = text.find("{")
    if start == -1:
        return ""

    end = find_matching_bracket(text, start, "{", "}")
    if end == -1:
        return ""

    # Extract the JSON string
    json_str = text[start : end + 1].strip()

    # Check if there's another JSON object right after
    next_start = text.find("{", end + 1)
    if next_start != -1:
        # If there is, make sure we're not in the middle of a string
        # by checking if the previous character is a quote
        if text[end:next_start].strip() and not text[end:next_start].strip().endswith('"'):
            # If not in a string, we've found a separate JSON object
            # and should stop at the first one
            return json_str

    return json_str


def extract_json_array(text: str) -> str:
    """
    Extract the first valid JSON array from the text.

    Args:
        text: The text to extract from

    Returns:
        str: The extracted JSON array string, or empty string if not found
    """
    start = text.find("[")
    if start == -1:
        return ""

    end = find_matching_bracket(text, start, "[", "]")
    if end == -1:
        return ""

    # Extract the JSON string
    json_str = text[start : end + 1].strip()

    # Check if there's another JSON array right after
    next_start = text.find("[", end + 1)
    if next_start != -1:
        # If there is, make sure we're not in the middle of a string
        # by checking if the previous character is a quote
        if text[end:next_start].strip() and not text[end:next_start].strip().endswith('"'):
            # If not in a string, we've found a separate JSON array
            # and should stop at the first one
            return json_str

    return json_str


def extract_code_block_content(text: str) -> str:
    """
    Extract content from a code block.

    Args:
        text: The text containing the code block

    Returns:
        str: The extracted content, or empty string if not found
    """
    if "```" not in text:
        return ""

    # Find the start and end of the first code block
    start = text.find("```")
    if start == -1:
        return ""

    # Check if it's a ```json block
    is_json_block = text[start : start + 7] == "```json"
    if is_json_block:
        start += 7
    else:
        start += 3

    # Find the end of the code block
    end = text.find("```", start)
    if end == -1:
        return ""

    return text[start:end].strip()


def llm_result2json(llm_str: str, expected_type: type[Dict | List] = dict) -> Union[Dict[str, Any], List[Any], None]:
    """
    Convert LLM output string to a JSON object or array.
    Supports the following formats:
    1. Plain JSON string
    2. Code block starting with ```json and ending with ```
    3. Code block starting with ``` and ending with ```

    Args:
        llm_str: String output from LLM
        expected_type: The expected type of the result (dict or list)

    Returns:
        Union[Dict[str, Any], List[Any], None]: JSON object/array on success, None on failure
    """
    try:
        cleaned_string = strip_json_str(llm_str)
        if not cleaned_string:
            return None  # Empty string should be treated as failure

        result = json_repair.loads(cleaned_string)

        # Ensure the result is of the expected type (dict or list)
        if not isinstance(result, (dict, list)):
            return None

        # If it's a dict, check if it has meaningful content
        if isinstance(result, dict):

            def _has_content(v):
                if v is None:
                    return False
                if isinstance(v, str):
                    return v.strip() != ""
                if isinstance(v, (list, dict)):
                    return len(v) > 0
                return True

            sql_content = result.get("sql")
            output_content = result.get("output")

            # Return None only if BOTH sql and output are empty/absent
            if not _has_content(sql_content) and not _has_content(output_content):
                return None

        return result

    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return None


def llm_result2sql(llm_str: str) -> Optional[str]:
    """
    Extract SQL from LLM output string.
    Looks for SQL in code blocks like ```sql ... ``` or ```SQL ... ```

    Args:
        llm_str: String output from LLM

    Returns:
        Optional[str]: Extracted SQL query if found, None if not found or invalid
    """
    try:
        if not llm_str or not isinstance(llm_str, str):
            return None

        # Look for SQL code blocks (case insensitive)
        import re

        # Pattern to match ```sql or ```SQL followed by content and ending ```
        sql_pattern = r"```(?:sql|SQL)\s*\n?(.*?)\n?```"
        match = re.search(sql_pattern, llm_str, re.DOTALL | re.IGNORECASE)

        if match:
            sql_content = match.group(1).strip()
            return sql_content if sql_content else None

        # Fallback: look for any code block that might contain SQL
        generic_pattern = r"```\s*\n?(.*?)\n?```"
        matches = re.findall(generic_pattern, llm_str, re.DOTALL)

        for code_block in matches:
            code_block = code_block.strip()
            if not code_block:
                continue

            # Simple heuristic: if it contains SQL keywords, consider it SQL
            sql_keywords = ["SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"]
            if any(keyword.lower() in code_block.lower() for keyword in sql_keywords):
                return code_block

        return None

    except (AttributeError, TypeError, ValueError):
        return None


def json_list2markdown_table(json_list: List[Dict[str, Any]]) -> str | None:
    """
    Convert a list of dictionaries to a markdown table format using tabulate.

    Args:
        json_list: List of dictionaries to convert

    Returns:
        str: Markdown formatted table string
    """
    if not json_list:
        return ""
    df = pd.DataFrame(json_list)
    return df.to_markdown()


def strip_json_str(llm_str: str) -> str:
    """
    Clean and extract JSON string from LLM output with robust handling.

    Handles:
    - Markdown code blocks (```json, ```)
    - Truncated JSON strings
    - JSON with embedded newlines
    - Malformed JSON that can be repaired

    Args:
        llm_str: Raw string from LLM

    Returns:
        str: Cleaned JSON string ready for parsing
    """
    if llm_str is None:
        return ""
    cleaned_string = llm_str.strip()
    if not cleaned_string:
        return ""

    # Handle markdown code blocks
    if cleaned_string.startswith("```json") and cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[len("```json") : -len("```")].strip()
    elif cleaned_string.startswith("```") and cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[len("```") : -len("```")].strip()

    # Try to extract JSON object/array if not already clean
    if not (cleaned_string.startswith("{") or cleaned_string.startswith("[")):
        # Look for JSON object
        start_obj = cleaned_string.find("{")
        if start_obj != -1:
            # Find matching closing brace
            end_obj = find_matching_bracket(cleaned_string, start_obj, "{", "}")
            if end_obj != -1:
                cleaned_string = cleaned_string[start_obj : end_obj + 1]
            else:
                # Truncated JSON object - try to find reasonable end
                cleaned_string = cleaned_string[start_obj:]
        else:
            # Look for JSON array
            start_arr = cleaned_string.find("[")
            if start_arr != -1:
                end_arr = find_matching_bracket(cleaned_string, start_arr, "[", "]")
                if end_arr != -1:
                    cleaned_string = cleaned_string[start_arr : end_arr + 1]
                else:
                    # Truncated JSON array
                    cleaned_string = cleaned_string[start_arr:]

    # Handle truncated JSON strings - if it ends abruptly, try to complete it
    if cleaned_string.startswith("{") and not cleaned_string.endswith("}"):
        # Count open braces vs closed braces
        open_braces = cleaned_string.count("{")
        close_braces = cleaned_string.count("}")
        missing_braces = open_braces - close_braces

        # Add missing closing braces if reasonable (not too many)
        if 0 < missing_braces <= 5:
            cleaned_string += "}" * missing_braces

    elif cleaned_string.startswith("[") and not cleaned_string.endswith("]"):
        # Count open brackets vs closed brackets
        open_brackets = cleaned_string.count("[")
        close_brackets = cleaned_string.count("]")
        missing_brackets = open_brackets - close_brackets

        # Add missing closing brackets if reasonable
        if 0 < missing_brackets <= 5:
            cleaned_string += "]" * missing_brackets

    # Handle incomplete string values - if we have an unclosed quote at the end
    if cleaned_string.count('"') % 2 == 1:
        # Find the last unclosed quote and check if it looks like an incomplete string
        last_quote = cleaned_string.rfind('"')
        if last_quote != -1:
            # Check if this quote starts a string value (not a key)
            before_quote = cleaned_string[:last_quote].rstrip()
            if before_quote.endswith(":") or before_quote.endswith(","):
                # This looks like an incomplete string value, close it
                cleaned_string += '"'

    return cleaned_string


def extract_json_str(llm_str: str) -> str:
    llm_str = llm_str.strip()
    if not llm_str:
        return ""

    json_str = llm_str
    if "```json" in llm_str:
        start = llm_str.index("```json")
        end = llm_str.rindex("```")
        json_str = llm_str[start + len("```json") : end]

    try:
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        if start >= 0 and end > start:
            return json_str[start:end]
    except Exception:
        pass
    return json_str


def load_jsonl(file_path) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_jsonl_iterator(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)
    return data


def load_jsonl_dict(file_path, key_field: str = "instance_id") -> Dict[str, Dict[str, Any]]:
    data = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            data[item[key_field]] = item
    return data


JSON_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def _normalize_for_json(data: Any) -> Any:
    """
    Convert many common Python/Pydantic/pandas/Numpy objects into JSON-serializable
    structures. This is a best-effort normalization intended for logging,
    transport, and display—not lossless persistence.
    """
    if isinstance(data, JSON_PRIMITIVE_TYPES):
        return data

    if isinstance(data, (datetime, date, time)):
        return data.isoformat()

    if isinstance(data, Decimal):
        return str(data)

    if isinstance(data, UUID):
        return str(data)

    if isinstance(data, Path):
        return str(data)

    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")

    if isinstance(data, Enum):
        enum_value = data.value
        return _normalize_for_json(enum_value) if enum_value is not data else data.name

    if BaseModel is not None and isinstance(data, BaseModel):
        try:
            model_data = data.model_dump(mode="json")  # type: ignore[arg-type]
        except TypeError:
            model_data = data.model_dump()  # type: ignore[arg-type]
        except AttributeError:  # Pydantic v1 fallback
            model_data = data.dict()
        return _normalize_for_json(model_data)

    if is_dataclass(data):
        return _normalize_for_json(asdict(data))

    if np is not None:
        if isinstance(data, np.generic):
            return _normalize_for_json(data.item())
        if isinstance(data, np.ndarray):
            return [_normalize_for_json(item) for item in data.tolist()]

    if isinstance(data, pd.DataFrame):
        return [_normalize_for_json(record) for record in data.replace({np.nan: None}).to_dict(orient="records")]

    if isinstance(data, pd.Series):
        return [_normalize_for_json(item) for item in data.tolist()]

    if isinstance(data, Mapping):
        return {str(key): _normalize_for_json(value) for key, value in data.items()}

    if isinstance(data, (list, tuple, set, frozenset)):
        return [_normalize_for_json(item) for item in data]

    if hasattr(data, "to_dict"):
        try:
            return _normalize_for_json(data.to_dict())
        except TypeError:
            pass

    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, bytearray)):
        return [_normalize_for_json(item) for item in data]

    if hasattr(data, "__dict__"):
        return _normalize_for_json(vars(data))

    return str(data)


def _dump_json(data: Any, *, indent: Optional[int] = None) -> str:
    """
    Shared implementation for serializing data to JSON, with best-effort support
    for a wide range of input types.
    """
    if isinstance(data, (str, bytes, bytearray)):
        text = data.decode("utf-8") if not isinstance(data, str) else data
        stripped = text.strip()
        if not stripped:
            return text
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return text

    normalized = _normalize_for_json(data)
    return json.dumps(normalized, ensure_ascii=False, indent=indent)


def to_pretty_str(json_data: Any) -> Optional[str]:
    """
    Serialize Python data to a human-readable JSON string.

    Args:
        json_data: Data to serialize. If None, returns None.

    Returns:
        A pretty-printed JSON string (2-space indentation) when input is not None,
        otherwise None.

    Notes:
        - Supports built-ins (dict/list/tuple/set), dataclasses, Enums, Decimal,
          datetime, pathlib.Path, UUID, pandas objects, NumPy arrays, iterables, and
          Pydantic BaseModel instances.
        - Custom objects fall back to `vars(obj)` when available or `str(obj)` as a
          last resort, so this helper is best-effort rather than lossless.
        - If a string/bytes payload already contains JSON, it is parsed and re-dumped.
          Otherwise it is returned verbatim, preserving non-JSON content.
        - `ensure_ascii=False` preserves Unicode characters as-is (e.g., Chinese),
          which is ideal for logs/UI, but may not be suitable for ASCII-only sinks.
        - Returning `None` (instead of an empty string) makes it easy to check for
          "no data", but callers must handle the Optional return type.
    """
    if json_data is None:
        return None
    return _dump_json(json_data, indent=2)


def to_str(json_data: Any) -> Optional[str]:
    """
    Serialize Python data to a compact JSON string.

    Args:
        json_data: Data to serialize. If None, returns None.

    Returns:
        A compact JSON string when input is not None, otherwise None.

    Notes:
        - Delegates to `_dump_json`, which normalizes many common non-JSON-native
          types (see `to_pretty_str` notes for details).
        - This does not sort keys or strip spaces; if you need the smallest payload,
          pass `separators=(",", ":")` at the call site after receiving this string.
        - `ensure_ascii=False` keeps Unicode characters intact; use `True` if the
          consumer expects ASCII-escaped output.
    """
    if json_data is None:
        return None
    return _dump_json(json_data)
