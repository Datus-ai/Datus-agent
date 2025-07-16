import io
import json
from typing import Any, Dict, List, Optional, Union

import json_repair
import pandas as pd


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


def llm_result2json(llm_str: str, expected_type: type[Dict | List] = dict) -> Union[Dict[str, Any], List[Any]]:
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
        Union[Dict[str, Any], List[Any]]: JSON object or array
    """
    cleaned_string = strip_json_str(llm_str)
    if not cleaned_string:
        return {} if expected_type == dict else []
    return json_repair.loads(cleaned_string)


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
    cleaned_string = llm_str.strip()
    if not cleaned_string:
        return ""

    if cleaned_string.startswith("```json") and cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[len("```json") : -len("```")].strip()
    elif cleaned_string.startswith("```") and cleaned_string.endswith("```"):
        cleaned_string = cleaned_string[len("```") : -len("```")].strip()

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
