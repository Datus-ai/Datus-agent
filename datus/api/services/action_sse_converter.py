"""
Public utilities for converting ActionHistory objects to SSE events.

Extracted from ChatTaskManager so that both streaming chat and
chat-history retrieval can share the same conversion logic.
"""

import json
from typing import Any, List, Optional, Set

from datus.agent.node.compact_archive import parse_archived_marker
from datus.api.models.cli_models import (
    IMessageContent,
    SSEDataType,
    SSEEvent,
    SSEMessageData,
    SSEMessagePayload,
    SSEUsageData,
    SSEUsageDelta,
)
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger
from datus.utils.time_utils import now_utc_iso, to_utc_iso

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


def _build_tool_call_content(
    action: ActionHistory, proxied_tool_names: Optional[Set[str]] = None
) -> List[IMessageContent]:
    """Build content for tool call started event.

    ``proxied_tool_names`` — names of tools the client must execute and
    report back (see ``apply_proxy_tools``). When provided, the payload
    carries ``proxied`` so the web client knows whether to run the tool
    (True) or the server already ran it (False). Omitted for converters
    without a live node (history retrieval): absent means the client
    falls back to its legacy heuristic.
    """
    function_name, arguments = _extract_function(action)
    payload_data = {
        "callToolId": action.action_id,
        "toolName": function_name,
        "toolParams": arguments,
    }
    if proxied_tool_names is not None:
        payload_data["proxied"] = function_name in proxied_tool_names
    return [IMessageContent(type="call-tool", payload=payload_data)]


def _build_tool_result_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for tool call completed event.

    When the tool failed, the payload includes an `error` field so the frontend
    can render the matching call-tool card in a failure state.
    """
    output = action.output

    start_time = action.start_time
    end_time = action.end_time
    duration = 0.0
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds()

    output_dict = output if isinstance(output, dict) else None
    short_desc = output_dict.get("summary", "") if output_dict else ""
    function_name, _ = _extract_function(action)
    result_payload = _normalize_tool_result_payload(
        output=output,
        status=action.status,
        fallback_error=action.messages,
    )

    payload_data = {
        "callToolId": action.action_id.removeprefix("complete_"),
        "toolName": function_name,
        "duration": duration,
        "shortDesc": short_desc,
        "result": result_payload,
    }

    error_message = result_payload.get("error")
    if error_message:
        payload_data["error"] = error_message

    return [IMessageContent(type="call-tool-result", payload=payload_data)]


def _normalize_tool_result_payload(
    *,
    output: Any,
    status: ActionStatus,
    fallback_error: str = "",
) -> dict[str, Any]:
    """Normalize backend tool output to the web-chatbot tool-result contract."""
    raw_output = output.get("raw_output", output) if isinstance(output, dict) else output
    raw_result = _parse_json_object(raw_output)

    success = status != ActionStatus.FAILED
    error = _extract_error(raw_result)
    direct_error = _extract_error(output)
    result = raw_result

    if _is_tool_result_envelope(raw_result):
        result = raw_result.get("result")
        raw_success = raw_result.get("success")
        if raw_success in (0, False) or error:
            success = False
        elif raw_success in (1, True):
            success = True

    if direct_error:
        success = False
        error = direct_error

    # Minor-compact archives a long tool output as the marker string; surface
    # the inline preview so the web UI does not render the raw
    # ``[DATUS_ARCHIVED] path=... preview=...`` text. The marker can land in
    # either the unwrapped ``result`` field (envelope case) or the bare
    # ``raw_result`` (non-envelope case where the whole output was replaced).
    marker = parse_archived_marker(result) or parse_archived_marker(raw_result)
    if marker is not None:
        result = {"archived": True, "preview": marker["preview"]}

    if status == ActionStatus.FAILED:
        success = False
        error = error or fallback_error or "Unknown error"

    payload: dict[str, Any] = {
        "success": 1 if success else 0,
        "result": result,
    }
    if error:
        payload["error"] = error
    return payload


def _parse_json_object(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return value
    return parsed if isinstance(parsed, dict) else value


def _is_tool_result_envelope(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "success" in value or "error" in value:
        return True
    return set(value) <= {"result"} and "result" in value


def _extract_error(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    return error.strip() if isinstance(error, str) and error.strip() else None


def _build_user_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for user message event."""
    input_data = action.input
    user_message = input_data.get("user_message", "") if isinstance(input_data, dict) else ""
    payload_data = {"content": user_message}
    return [IMessageContent(type="markdown", payload=payload_data)]


def _build_response_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for final response event."""
    contents = []
    action_output = action.output if isinstance(action.output, dict) else {}
    if "sql" in action_output and action_output["sql"]:
        sql = action_output.get("sql")
        sql_payload = {"codeType": "sql", "content": sql}
        contents.append(IMessageContent(type="code", payload=sql_payload))

    response = action_output.get("response") or action_output.get("content") or action_output.get("raw_output") or ""
    if response:
        resp_payload = {"content": str(response)}
        contents.append(IMessageContent(type="markdown", payload=resp_payload))
    return contents


def _is_plain_assistant_response(action: ActionHistory) -> bool:
    """Return True for completed assistant text that should render as markdown."""
    if action.role != ActionRole.ASSISTANT or action.status != ActionStatus.SUCCESS:
        return False
    if action.action_type != "response":
        return False
    output = action.output if isinstance(action.output, dict) else {}
    return output.get("is_thinking") is False


def _build_thinking_content(action: ActionHistory) -> Optional[List[IMessageContent]]:
    """Extract text content from action for markdown display."""
    from datus.utils.text_utils import strip_litellm_placeholder

    action_type = action.action_type

    if action_type == "llm_generation":
        messages = strip_litellm_placeholder(action.messages)
        return [IMessageContent(type="thinking", payload={"content": messages})] if messages else None

    output = action.output
    content = None
    if output and isinstance(output, dict):
        for key in ["response", "raw_output", "output", "thinking", "content"]:
            if key in output and output[key]:
                candidate = strip_litellm_placeholder(str(output[key]))
                if candidate:
                    content = candidate
                    break

    if not content:
        messages = strip_litellm_placeholder(action.messages)
        return [IMessageContent(type="thinking", payload={"content": messages})] if messages else None

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
        if "explanation" in result_json and result_json["explanation"]:
            resp_payload = {"content": result_json.get("explanation", "")}
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
    from datus.schemas.interaction_event import InteractionEvent

    events = InteractionEvent.from_broker_input(action.input if isinstance(action.input, dict) else {})

    requests_payload = []
    for ev in events:
        options = [{"key": k, "title": v} for k, v in ev.choices.items()] if ev.choices else None
        requests_payload.append(
            {
                "title": ev.title,
                "content": ev.content,
                "options": options,
                "defaultChoice": ev.default_choice,
                "contentType": ev.content_type,
                "allowFreeText": ev.allow_free_text,
                "multiSelect": ev.multi_select,
            }
        )

    payload_data = {
        "interactionKey": action.action_id,
        "actionType": action.action_type,
        "requests": requests_payload,
    }

    return [IMessageContent(type="user-interaction", payload=payload_data)]


_ARTIFACT_PREVIEW_LIMIT = 200


def _build_artifact_content(action: ActionHistory) -> Optional[List[IMessageContent]]:
    """Build an artifact-card content block from a finished gen_visual_* run.

    Returns ``None`` when the action does not actually carry an artifact
    payload (missing slug or kind) so the caller can fall back to the
    regular ``_response`` markdown rendering.

    The payload mirrors :class:`IArtifactPayload`. ``mode`` reuses the
    backend artifact-tools vocabulary (``'new'`` / ``'edit'``) so there
    is no translation layer between the NodeResult and the wire payload.
    """
    output = action.output if isinstance(action.output, dict) else {}
    slug = output.get("report_slug") or output.get("dashboard_slug")
    kind = output.get("artifact_kind")
    if not slug or kind not in ("report", "dashboard"):
        return None

    response_text = output.get("response") or ""
    preview = response_text.strip() if isinstance(response_text, str) else ""
    if len(preview) > _ARTIFACT_PREVIEW_LIMIT:
        preview = preview[:_ARTIFACT_PREVIEW_LIMIT].rstrip() + "…"

    payload: dict[str, Any] = {
        "slug": slug,
        "kind": kind,
        "mode": output.get("artifact_mode"),
        "name": output.get("name"),
        "description": output.get("description"),
        "created_at": output.get("created_at"),
        "preview_summary": preview or None,
    }
    return [IMessageContent(type="artifact", payload=payload)]


def _build_subagent_complete_content(action: ActionHistory) -> List[IMessageContent]:
    """Build content for sub-agent completion summary event.

    When the sub-agent failed, the payload includes an `error` field so the
    frontend can render the matching subagent card in a failure state.
    """
    output = action.output if isinstance(action.output, dict) else {}
    duration = (action.end_time - action.start_time).total_seconds() if action.start_time and action.end_time else 0.0
    payload_data = {
        "subagentType": output.get("subagent_type", "unknown"),
        "toolCount": output.get("tool_count", 0),
        "duration": duration,
    }

    error_message = output.get("error")
    if not error_message and action.status == ActionStatus.FAILED:
        error_message = action.messages or "Unknown error"
    if error_message:
        payload_data["error"] = error_message

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


def _build_token_usage_event(action: ActionHistory, event_id: int) -> Optional[SSEEvent]:
    """Convert a ``token_usage`` action into a dedicated ``usage`` SSE event.

    Producer side: :class:`TokenUsageHook` records turn-cumulative usage and
    a per-call delta into ``action.output``. We pass both through so the
    frontend can render the running total and the single-call contribution
    without re-aggregating on its end.
    """
    output = action.output if isinstance(action.output, dict) else {}
    cumulative = output.get("cumulative") if isinstance(output.get("cumulative"), dict) else {}
    delta = output.get("delta") if isinstance(output.get("delta"), dict) else {}
    try:
        context_length = int(output.get("context_length", 0) or 0)
    except (TypeError, ValueError):
        context_length = 0
    try:
        last_call_input_tokens = int(output.get("last_call_input_tokens", 0) or 0)
    except (TypeError, ValueError):
        last_call_input_tokens = 0

    def _i(d: dict, key: str) -> int:
        try:
            return int(d.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    sse_data = SSEUsageData(
        # ``session_id`` is populated by :class:`ChatTaskManager` (the only
        # caller that has the service-level id). We leave it as the action's
        # session-scoped identifier when unavailable so the event still
        # carries a recognizable correlation key.
        session_id=str(getattr(action, "action_id", "") or ""),
        # ``agent_session_id`` is the session of the node that produced this
        # usage — for a forwarded sub-agent action that is the sub-agent's own
        # session, not the parent's. ChatTaskManager only overrides this for
        # main-agent (depth==0) usage.
        llm_session_id=output.get("agent_session_id"),
        # ``depth``/``parent_action_id`` let API consumers separate main-agent
        # usage (depth==0) from sub-agent usage (depth>0) and group the latter
        # under the originating ``task()`` call.
        depth=int(getattr(action, "depth", 0) or 0),
        parent_action_id=getattr(action, "parent_action_id", None),
        requests=_i(cumulative, "requests"),
        input_tokens=_i(cumulative, "input_tokens"),
        output_tokens=_i(cumulative, "output_tokens"),
        total_tokens=_i(cumulative, "total_tokens"),
        cached_tokens=_i(cumulative, "cached_tokens"),
        reasoning_tokens=_i(cumulative, "reasoning_tokens"),
        last_call_input_tokens=last_call_input_tokens,
        context_length=context_length,
        delta=SSEUsageDelta(
            requests=_i(delta, "requests"),
            input_tokens=_i(delta, "input_tokens"),
            output_tokens=_i(delta, "output_tokens"),
            total_tokens=_i(delta, "total_tokens"),
            cached_tokens=_i(delta, "cached_tokens"),
            reasoning_tokens=_i(delta, "reasoning_tokens"),
        ),
    )
    return SSEEvent(
        id=event_id,
        event="usage",
        data=sse_data,
        timestamp=to_utc_iso(getattr(action, "start_time", None)) or now_utc_iso(),
    )


def action_to_sse_event(
    action: ActionHistory,
    event_id: int,
    message_id: str,
    include_user_message: bool = False,
    stream_thinking: bool = False,
    is_first_delta: bool = True,
    is_update: bool = False,
    include_final_response: bool = False,
    proxied_tool_names: Optional[Set[str]] = None,
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
    stream_thinking : bool
        If True, thinking_delta actions are emitted as incremental SSE events.
        If False, thinking_delta actions are silently skipped.
    is_first_delta : bool
        If True, the first thinking_delta uses CREATE_MESSAGE; subsequent deltas
        use APPEND_MESSAGE. Only relevant when stream_thinking=True.
    is_update : bool
        If True, the event uses UPDATE_MESSAGE to overwrite previously streamed
        deltas with the complete thinking response.
    include_final_response : bool
        If True, convert node wrapper actions such as chat_response to markdown.
        Streaming callers should only set this when no plain assistant response
        has already been emitted for the turn.
    proxied_tool_names : Optional[Set[str]]
        Tool names proxied to the client for execution; stamps ``proxied``
        on call-tool payloads (see ``_build_tool_call_content``). Pass None
        when no live node is available (history retrieval).
    """
    try:
        role = action.role
        status = action.status

        sse_role = "assistant"
        sse_type = SSEDataType.CREATE_MESSAGE

        if stream_thinking and is_update:
            sse_type = SSEDataType.UPDATE_MESSAGE

        if action.action_type == "token_usage":
            return _build_token_usage_event(action, event_id)

        if action.action_type == "compact_progress":
            return None  # REPL-only in-progress hint; not surfaced over SSE

        if action.action_type == "thinking_delta":
            if not stream_thinking:
                return None
            output = action.output if isinstance(action.output, dict) else {}
            delta_text = output.get("delta", "")
            if not is_first_delta:
                sse_type = SSEDataType.APPEND_MESSAGE
            contents = [IMessageContent(type="thinking", payload={"content": delta_text})]
        elif action.action_type == "finalize_progress":
            # Visual-artifact finalize streams 3 stage transitions through
            # a single bubble. All 3 actions share an action_id (= wire
            # message_id) so the chat panel finds the existing message and
            # the SSE converter alternates CREATE_MESSAGE -> UPDATE_MESSAGE
            # -> UPDATE_MESSAGE. ``is_update`` is set by the task_manager
            # loop after the first emission with this action_id.
            output = action.output if isinstance(action.output, dict) else {}
            text = output.get("text", "")
            if not text:
                return None
            contents = [IMessageContent(type="markdown", payload={"content": text})]
            sse_type = SSEDataType.UPDATE_MESSAGE if is_update else SSEDataType.CREATE_MESSAGE
        elif action.action_type == "compact_summary":
            # Post-compact summary: a markdown bubble carrying a ``kind`` marker
            # so the frontend can recognise the compacted-context summary (and
            # e.g. collapse the prior transcript). The backend never clears
            # anything itself — clearing is a REPL-only concern.
            output = action.output if isinstance(action.output, dict) else {}
            summary = str(output.get("summary", "") or "")
            if not summary:
                return None
            # ``history_jsonl`` is a server-local filesystem path that is unusable
            # for remote clients and would disclose internal server layout, so it
            # is deliberately omitted from the SSE-facing payload.
            contents = [
                IMessageContent(
                    type="markdown",
                    payload={
                        "content": summary,
                        "kind": "compact_summary",
                        "summary_token": int(output.get("summary_token", 0) or 0),
                    },
                )
            ]
            sse_type = SSEDataType.CREATE_MESSAGE
        elif action.action_type == SUBAGENT_COMPLETE_ACTION_TYPE:
            contents = _build_subagent_complete_content(action)
        elif role == ActionRole.TOOL and status == ActionStatus.PROCESSING:
            contents = _build_tool_call_content(action, proxied_tool_names)
        elif role == ActionRole.TOOL:
            contents = _build_tool_result_content(action)
        elif role == ActionRole.INTERACTION and status == ActionStatus.PROCESSING:
            contents = _build_interaction_content(action)
        elif role == ActionRole.INTERACTION and status == ActionStatus.SUCCESS:
            contents = _build_interaction_result_content(action)
            if contents is None:
                return None
        elif role == ActionRole.USER:
            # ``user_insert`` is a mid-run injection: text the user typed
            # into the TUI / POSTed to ``/chat/insert`` while the agent
            # was already streaming. It must always reach the SSE client,
            # independent of ``include_user_message`` (which gates the
            # initial-request action that the caller already knows about).
            if include_user_message or action.action_type == "user_insert":
                contents = _build_user_content(action)
                sse_role = "user"
            else:
                return None
        elif (
            role == ActionRole.ASSISTANT and status == ActionStatus.SUCCESS and action.action_type.endswith("_response")
        ):
            # gen_visual_report / gen_visual_dashboard completions ship an
            # ``artifact_kind`` on the output — try to render an artifact
            # card the frontend can open directly. The card fires
            # regardless of ``include_final_response`` because it is not a
            # substitute for the LLM's prose (already streamed earlier as
            # a separate response action); it is purely additive.
            contents = None
            if isinstance(action.output, dict) and action.output.get("artifact_kind") in ("report", "dashboard"):
                contents = _build_artifact_content(action)

            # Either this is a plain wrapper response (no artifact) or the
            # artifact payload was malformed (e.g. missing slug). Don't
            # drop the event silently — fall back to the standard
            # ``_response`` handling so the assistant's parsed output can
            # still be surfaced to history tooling when requested.
            if contents is None:
                if not include_final_response:
                    return None  # ignore parsed final response
                contents = _build_response_content(action)
                if not contents:
                    return None
        elif _is_plain_assistant_response(action):
            contents = _build_response_content(action)
            if not contents:
                return None
        elif status == ActionStatus.FAILED:
            contents = _build_error_content(action)
        else:
            contents = _build_thinking_content(action)
            if contents is None:
                return None  # Skip empty content

        sse_data = SSEMessageData(
            type=sse_type,
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
            timestamp=to_utc_iso(getattr(action, "start_time", None)) or now_utc_iso(),
        )

    except Exception as e:
        logger.error(f"Error converting action to SSE event: {str(e)}")
        return None
