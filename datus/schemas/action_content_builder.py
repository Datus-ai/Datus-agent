# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Pure functions for converting ActionHistory objects to MessageContent lists.

Extracted from ChatService to be shared between API SSE streaming and CLI print mode.
"""

import json
from typing import List, Optional

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.schemas.message_content import MessageContent
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def extract_function(action: ActionHistory) -> tuple[str, dict]:
    """Extract function name and arguments from action.input."""
    input_data = action.input
    if not isinstance(input_data, dict):
        return "unknown", {}

    function_name = input_data.get("function_name", "unknown")
    arguments = input_data.get("arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}

    if not isinstance(arguments, dict):
        arguments = {}

    return function_name, arguments


def build_tool_call_content(action: ActionHistory) -> List[MessageContent]:
    """Build content for tool call started event."""
    function_name, arguments = extract_function(action)
    payload_data = {
        "callToolId": action.action_id,
        "toolName": function_name,
        "toolParams": arguments,
    }
    return [MessageContent(type="call-tool", payload=payload_data)]


def build_tool_result_content(action: ActionHistory) -> List[MessageContent]:
    """Build content for tool call completed event."""
    output = action.output

    start_time = action.start_time
    end_time = action.end_time
    duration = 0.0
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds()

    is_dict_output = isinstance(output, dict)
    short_desc = output.get("summary", "") if is_dict_output else ""
    function_name, _ = extract_function(action)

    payload_data = {
        "callToolId": action.action_id.removeprefix("complete_"),
        "toolName": function_name,
        "duration": duration,
        "shortDesc": short_desc,
        "result": output.get("raw_output", output) if is_dict_output else output,
    }
    return [MessageContent(type="call-tool-result", payload=payload_data)]


def build_thinking_content(action: ActionHistory) -> Optional[List[MessageContent]]:
    """Extract text content from action for thinking/markdown display."""
    action_type = action.action_type

    if action_type == "llm_generation":
        return [MessageContent(type="thinking", payload={"content": action.messages})]

    output = action.output
    content = None
    if output and isinstance(output, dict):
        for key in ["response", "raw_output", "output", "thinking", "content"]:
            if key in output and output[key]:
                content = str(output[key])
                break

    if not content:
        return [MessageContent(type="thinking", payload={"content": action.messages})]

    result_json = llm_result2json(content)

    if result_json:
        contents = []
        if "sql" in result_json and result_json["sql"]:
            sql = result_json.get("sql")
            sql_payload = {"codeType": "sql", "content": sql}
            contents.append(MessageContent(type="code", payload=sql_payload))
        if "output" in result_json and result_json["output"]:
            resp_payload = {"content": result_json.get("output", "")}
            contents.append(MessageContent(type="markdown", payload=resp_payload))

        if contents:
            return contents

    return [MessageContent(type="thinking", payload={"content": content})]


def build_interaction_content(action: ActionHistory) -> List[MessageContent]:
    """Build content for user interaction event (PROCESSING status)."""
    input_data = action.input if isinstance(action.input, dict) else {}

    content = input_data.get("content", "")
    choices = input_data.get("choices", {})

    options = None
    if choices and isinstance(choices, dict):
        options = [{"key": key, "title": title} for key, title in choices.items()]

    payload_data = {
        "interactionKey": action.action_id,
        "content": content,
        "options": options,
    }
    return [MessageContent(type="user-interaction", payload=payload_data)]


def build_interaction_result_content(action: ActionHistory) -> Optional[List[MessageContent]]:
    """Build content for interaction result event (SUCCESS status)."""
    output = action.output if isinstance(action.output, dict) else {}
    content = output.get("content", "")
    if not content:
        return None
    payload_data = {"content": content}
    return [MessageContent(type="markdown", payload=payload_data)]


def build_response_content(action: ActionHistory) -> List[MessageContent]:
    """Build content for final response event."""
    contents = []
    action_output = action.output
    if "sql" in action_output and action_output["sql"]:
        sql = action_output.get("sql")
        sql_payload = {"codeType": "sql", "content": sql}
        contents.append(MessageContent(type="code", payload=sql_payload))

    resp_payload = {"content": action_output.get("response", "")}
    contents.append(MessageContent(type="markdown", payload=resp_payload))
    return contents


def build_user_content(action: ActionHistory) -> List[MessageContent]:
    """Build content for user message event."""
    input_data = action.input
    user_message = input_data.get("user_message", "") if isinstance(input_data, dict) else ""
    payload_data = {"content": user_message}
    return [MessageContent(type="markdown", payload=payload_data)]


def action_to_content(action: ActionHistory) -> Optional[List[MessageContent]]:
    """Convert an ActionHistory object to a list of MessageContent.

    Routes to the appropriate build function based on role/status.
    Returns None if the action should be skipped.
    """
    role = action.role
    status = action.status

    if role == ActionRole.TOOL and status == ActionStatus.PROCESSING:
        return build_tool_call_content(action)
    elif role == ActionRole.TOOL:
        return build_tool_result_content(action)
    elif role == ActionRole.INTERACTION and status == ActionStatus.PROCESSING:
        return build_interaction_content(action)
    elif role == ActionRole.INTERACTION and status == ActionStatus.SUCCESS:
        return build_interaction_result_content(action)
    elif role == ActionRole.USER:
        return None
    elif (
        role == ActionRole.ASSISTANT
        and status == ActionStatus.SUCCESS
        and action.action_type
        and action.action_type.endswith("_response")
    ):
        return None  # ignore parsed final response
    else:
        return build_thinking_content(action)
