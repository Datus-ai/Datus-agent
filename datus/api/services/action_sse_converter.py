"""
Public utilities for converting ActionHistory objects to SSE events.

Extracted from ChatTaskManager so that both streaming chat and
chat-history retrieval can share the same conversion logic.
"""

import json
from datetime import datetime
from typing import List, Optional

from datus.api.models.cli_models import (
    IMessageContent,
    SSEDataType,
    SSEEvent,
    SSEMessageData,
    SSEMessagePayload,
)
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Helper builders
# ------------------------------------------------------------------


def _extract_function(action: ActionHistory) -> tuple[str, dict]:
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


def _build_tool_call_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for tool call started event."""
    function_name, arguments = _extract_function(action)
    payload_data = {
        "callToolId": action.action_id,
        "toolName": function_name,
        "toolParams": arguments,
    }
    return [IMessageContent(type="call-tool", payload=payload_data)]


def _build_tool_result_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for tool call completed event."""
    output = action.output

    start_time = action.start_time
    end_time = action.end_time
    duration = 0.0
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds()

    output_dict = output if isinstance(output, dict) else None
    short_desc = output_dict.get("summary", "") if output_dict else ""
    function_name, _ = _extract_function(action)

    payload_data = {
        "callToolId": action.action_id.removeprefix("complete_"),
        "toolName": function_name,
        "duration": duration,
        "shortDesc": short_desc,
        "result": output_dict.get("raw_output", output) if output_dict else output,
    }
    return [IMessageContent(type="call-tool-result", payload=payload_data)]


def _build_user_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for user message event."""
    input_data = action.input
    user_message = input_data.get("user_message", "") if isinstance(input_data, dict) else ""
    payload_data = {"content": user_message}
    return [IMessageContent(type="markdown", payload=payload_data)]


def _build_response_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for final response event."""
    contents = []
    action_output = action.output
    if "sql" in action_output and action_output["sql"]:
        sql = action_output.get("sql")
        sql_payload = {"codeType": "sql", "content": sql}
        contents.append(IMessageContent(type="code", payload=sql_payload))

    resp_payload = {"content": action_output.get("response", "")}
    contents.append(IMessageContent(type="markdown", payload=resp_payload))
    return contents


def _build_thinking_content(action: ActionHistory) -> Optional[List[IMessageContent]]:
    """Extract text content from action for markdown display."""
    action_type = action.action_type

    if action_type == "llm_generation":
        return [IMessageContent(type="thinking", payload={"content": action.messages})]

    output = action.output
    content = None
    if output and isinstance(output, dict):
        for key in ["response", "raw_output", "output", "thinking", "content"]:
            if key in output and output[key]:
                content = str(output[key])
                break

    if not content:
        return [IMessageContent(type="thinking", payload={"content": action.messages})]

    result_json = llm_result2json(content)

    if result_json:
        contents = []
        if "sql" in result_json and result_json["sql"]:
            sql = result_json.get("sql")
            sql_payload = {"codeType": "sql", "content": sql}
            contents.append(IMessageContent(type="code", payload=sql_payload))
        if "output" in result_json and result_json["output"]:
            resp_payload = {"content": result_json.get("output", "")}
            contents.append(IMessageContent(type="markdown", payload=resp_payload))

        if contents:
            return contents

    return [IMessageContent(type="thinking", payload={"content": content})]


def _build_error_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for failed action event, extracting error from BaseResult format."""
    output = action.output if isinstance(action.output, dict) else {}
    error_message = output.get("error") or action.messages or "Unknown error"
    return [IMessageContent(type="error", payload={"content": error_message})]


def _build_interaction_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for user interaction event (PROCESSING status)."""
    input_data = action.input if isinstance(action.input, dict) else {}

    contents = input_data.get("contents", [])
    choices_list = input_data.get("choices", [])
    default_choices = input_data.get("default_choices", [])
    content_type = input_data.get("content_type", "markdown")
    allow_free_text = input_data.get("allow_free_text", False)

    requests_payload = []
    for i, content in enumerate(contents):
        choices = choices_list[i] if i < len(choices_list) else {}
        options = [{"key": k, "title": v} for k, v in choices.items()] if choices else None
        default_choice = default_choices[i] if i < len(default_choices) else ""
        requests_payload.append(
            {
                "content": content,
                "options": options,
                "defaultChoice": default_choice,
                "contentType": content_type,
                "allowFreeText": allow_free_text,
            }
        )

    payload_data = {
        "interactionKey": action.action_id,
        "actionType": action.action_type,
        "requests": requests_payload,
    }

    return [IMessageContent(type="user-interaction", payload=payload_data)]


def _build_subagent_complete_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for sub-agent completion summary event."""
    output = action.output if isinstance(action.output, dict) else {}
    duration = (action.end_time - action.start_time).total_seconds() if action.start_time and action.end_time else 0.0
    payload_data = {
        "subagentType": output.get("subagent_type", "unknown"),
        "toolCount": output.get("tool_count", 0),
        "duration": duration,
    }
    return [IMessageContent(type="subagent-complete", payload=payload_data)]


def _build_interaction_result_content(action: ActionHistory) -> Optional[List[IMessageContent]]:
    """Build content for interaction result event (SUCCESS status)."""
    output = action.output if isinstance(action.output, dict) else {}
    content = output.get("content", "")
    if not content:
        return None
    payload_data = {"content": content}
    return [IMessageContent(type="markdown", payload=payload_data)]


# ------------------------------------------------------------------
# Public converter
# ------------------------------------------------------------------


def action_to_sse_event(
    action: ActionHistory,
    event_id: int,
    message_id: str,
    include_user_message: bool = False,
) -> Optional[SSEEvent]:
    """Convert an ActionHistory object to an SSEEvent.

    Parameters
    ----------
    action : ActionHistory
        The action to convert.
    event_id : int
        Sequential event identifier.
    message_id : str
        Unique message identifier.
    include_user_message : bool
        If True, USER-role actions are converted to SSE events (for chat history).
        If False, USER-role actions return None (for streaming).
    """
    try:
        role = action.role
        status = action.status

        sse_role = "assistant"

        if status == ActionStatus.FAILED:
            contents = _build_error_content(action)
        elif action.action_type == SUBAGENT_COMPLETE_ACTION_TYPE:
            contents = _build_subagent_complete_content(action)
        elif role == ActionRole.TOOL and status == ActionStatus.PROCESSING:
            contents = _build_tool_call_content(action)
        elif role == ActionRole.TOOL:
            contents = _build_tool_result_content(action)
        elif role == ActionRole.INTERACTION and status == ActionStatus.PROCESSING:
            contents = _build_interaction_content(action)
        elif role == ActionRole.INTERACTION and status == ActionStatus.SUCCESS:
            contents = _build_interaction_result_content(action)
            if contents is None:
                return None
        elif role == ActionRole.USER:
            if include_user_message:
                contents = _build_user_content(action)
                sse_role = "user"
            else:
                return None
        elif (
            role == ActionRole.ASSISTANT and status == ActionStatus.SUCCESS and action.action_type.endswith("_response")
        ):
            return None  # ignore parsed final response
        else:
            contents = _build_thinking_content(action)
            if contents is None:
                return None  # Skip empty content

        sse_data = SSEMessageData(
            type=SSEDataType.CREATE_MESSAGE,
            payload=SSEMessagePayload(
                message_id=message_id,
                role=sse_role,
                content=contents,
                depth=action.depth,
                parent_action_id=action.parent_action_id,
            ),
        )

        return SSEEvent(
            id=event_id,
            event="message",
            data=sse_data,
            timestamp=getattr(action, "start_time", datetime.now()).isoformat() + "Z",
        )

    except Exception as e:
        logger.error(f"Error converting action to SSE event: {str(e)}")
        return None
