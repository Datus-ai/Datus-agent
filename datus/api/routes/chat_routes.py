"""
API routes for Chat endpoints.

All business logic (session queries, history retrieval) is delegated to
ChatService via DatusService. Routes are thin wrappers that handle HTTP
concerns only.

Streaming endpoints use the project-scoped ChatTaskManager (via
DatusService.task_manager) to run the agentic loop in a background
asyncio.Task so that client disconnects do not cancel the computation.
"""

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse

from datus.api.constants import BUILTIN_SUBAGENTS
from datus.api.deps import AppContextDep, ServiceDep
from datus.api.hooks import (
    ChatHooks,
    ChatPostUsageContext,
    ChatPreCheckOutcome,
    get_chat_hooks,
)
from datus.api.models.base_models import Result
from datus.api.models.chat_models import (
    InsertMessageData,
    InsertMessageInput,
    ResumeChatInput,
    StopChatInput,
    ToolResultData,
    ToolResultInput,
)
from datus.api.models.cli_models import (
    ChatHistoryData,
    ChatSessionData,
    CompactSessionData,
    CompactSessionInput,
    FeedbackChatInput,
    SSEErrorData,
    SSEEvent,
    StreamChatInput,
    UserInteractionInput,
)
from datus.tools.data_access_policy import DataAccessConfig
from datus.utils.feedback_prompt import build_reaction_feedback_prompt
from datus.utils.loggings import get_logger
from datus.utils.time_utils import now_utc_iso

logger = get_logger(__name__)

if TYPE_CHECKING:
    from datus.api.auth.context import AppContext
    from datus.api.services.datus_service import DatusService

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# Additional builtin subagents accepted by ``stream_chat`` beyond the canonical
# ``BUILTIN_SUBAGENTS`` set — these are wired directly in
# ``ChatTaskManager._create_node`` but are not listed as user-creatable agents
# in ``datus.api.constants``. Keep this list in sync with the dispatch branches
# in :meth:`ChatTaskManager._create_node`.
_EXTRA_BUILTIN_SUBAGENTS = {"feedback"}


def _is_valid_subagent_id(svc, subagent_id: str) -> bool:
    """Return True if *subagent_id* resolves to a builtin or custom sub-agent."""
    if subagent_id in BUILTIN_SUBAGENTS or subagent_id in _EXTRA_BUILTIN_SUBAGENTS:
        return True
    agentic_nodes = getattr(svc.agent_config, "agentic_nodes", None) or {}
    if subagent_id in agentic_nodes:
        return True
    # Custom sub-agents may be keyed by sanitized node_name with the original
    # UUID id stored under "id" — match either form.
    for entry in agentic_nodes.values():
        if isinstance(entry, dict) and entry.get("id") == subagent_id:
            return True
    return False


def _data_access_principal_pre_check(svc: "DatusService", ctx: "AppContext") -> Optional[ChatPreCheckOutcome]:
    """Fail fast when enabled data-access policies need missing principal fields."""
    agent_config = getattr(svc, "agent_config", None)
    data_access_config = getattr(agent_config, "data_access_config", None)
    if not isinstance(data_access_config, DataAccessConfig) or not data_access_config.enabled:
        return None

    principal = getattr(ctx, "principal", None) or {}
    required_paths = _required_principal_paths(data_access_config.raw)
    missing_paths = _missing_principal_paths(principal, required_paths)
    if not missing_paths:
        return None

    detail = ""
    if missing_paths:
        missing_fields = ", ".join(f"principal.{path}" for path in missing_paths)
        detail = f" Missing principal field(s): {missing_fields}."
    return ChatPreCheckOutcome(
        allow=False,
        error=(
            "Data access is enabled, but this request is missing principal data required by policy."
            f"{detail} "
            "Authenticate the request with a provider that populates principal fields required by "
            "agent.data_access policies. The agent cannot infer or set request principal from SQL."
        ),
        error_type="DATA_ACCESS_PRINCIPAL_REQUIRED",
    )


def _required_principal_paths(raw: Any) -> list[str]:
    paths: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value_from = value.get("value_from")
            if isinstance(value_from, str) and value_from.startswith("principal."):
                path = value_from[len("principal.") :].strip()
                if path:
                    paths.add(path)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw)
    return sorted(paths)


def _missing_principal_paths(principal: dict[str, Any], required_paths: list[str]) -> list[str]:
    return [path for path in required_paths if not _principal_path_exists(principal, path)]


def _principal_path_exists(principal: dict[str, Any], path: str) -> bool:
    current: Any = principal
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return current not in (None, "", [])


# ========== Stream Chat ==========


@router.post(
    "/stream",
    summary="Stream Chat Message",
    description="Send chat message with streaming response (Server-Sent Events). "
    "Set subagent_id to route to a specific sub-agent.",
)
async def stream_chat(
    request: StreamChatInput,
    svc: ServiceDep,
    ctx: AppContextDep,
    http_request: Request,
):
    sub_agent_id = request.subagent_id
    if sub_agent_id and not _is_valid_subagent_id(svc, sub_agent_id):
        raise HTTPException(
            status_code=404,
            detail=f"Subagent '{sub_agent_id}' not found",
        )

    data_access_denial = _data_access_principal_pre_check(svc, ctx)
    if data_access_denial:
        return StreamingResponse(
            _emit_pre_check_denial(request, data_access_denial),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    hooks = get_chat_hooks()
    pre_outcome = await _run_pre_chat_hook(hooks, http_request, request, ctx.user_id)
    if pre_outcome and not pre_outcome.allow:
        return StreamingResponse(
            _emit_pre_check_denial(request, pre_outcome),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    pre_extra = pre_outcome.extra if pre_outcome else {}

    async def generate_sse():
        async for chunk in _stream_with_post_hook(
            svc.chat.stream_chat(request, sub_agent_id=sub_agent_id, user_id=ctx.user_id, principal=ctx.principal),
            http_request=http_request,
            request=request,
            user_id=ctx.user_id,
            hooks=hooks,
            pre_extra=pre_extra,
        ):
            yield chunk

    return StreamingResponse(generate_sse(), media_type="text/event-stream", headers=_sse_headers())


# ========== Reaction-triggered Feedback ==========


@router.post(
    "/feedback",
    summary="Stream Feedback Agent (reaction-triggered)",
    description=(
        "Trigger the feedback agent from a reaction event (IM emoji, UI thumbs, etc.). "
        "The server builds the canonical user prompt from reaction_emoji/reference_msg/reaction_msg "
        "and routes the request to the feedback sub-agent, which copies the source session and archives reusable knowledge."
    ),
)
async def stream_chat_feedback(
    request: FeedbackChatInput,
    svc: ServiceDep,
    ctx: AppContextDep,
):
    rendered_message = build_reaction_feedback_prompt(
        reaction_emoji=request.reaction_emoji,
        reference_msg=request.reference_msg,
        reaction_msg=request.reaction_msg,
    )
    stream_input = StreamChatInput(
        **request.model_dump(
            exclude={"message", "reaction_emoji", "reference_msg", "reaction_msg"},
        ),
        message=rendered_message,
        subagent_id="feedback",
    )
    data_access_denial = _data_access_principal_pre_check(svc, ctx)
    if data_access_denial:
        return StreamingResponse(
            _emit_pre_check_denial(stream_input, data_access_denial),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    async def generate_sse():
        async for event in svc.chat.stream_chat(
            stream_input, sub_agent_id="feedback", user_id=ctx.user_id, principal=ctx.principal
        ):
            yield f"id: {event.id}\nevent: {event.event}\ndata: {event.data.model_dump_json()}\n\n"

    return StreamingResponse(generate_sse(), media_type="text/event-stream", headers=_sse_headers())


# ========== Resume Chat ==========


@router.post(
    "/resume",
    summary="Resume Chat Session",
    description="Reconnect to a running chat task and consume events from a given cursor",
)
async def resume_chat(
    request: ResumeChatInput,
    svc: ServiceDep,
):
    task_manager = svc.task_manager
    task = task_manager.get_task(request.session_id)
    if task is None:
        return Result[dict](
            success=False,
            errorCode="TASK_NOT_FOUND",
            errorMessage="Task not found or already completed. Use the history API to retrieve messages.",
        )

    async def generate_sse():
        async for event in task_manager.consume_events(task, start_from=request.from_event_id):
            yield f"id: {event.id}\nevent: {event.event}\ndata: {event.data.model_dump_json()}\n\n"

    return StreamingResponse(generate_sse(), media_type="text/event-stream", headers=_sse_headers())


# ========== Stop Chat ==========


@router.post(
    "/stop",
    response_model=Result[dict],
    summary="Stop Chat Session",
    description="Stop a currently running chat session",
)
async def stop_chat(
    request: StopChatInput,
    svc: ServiceDep,
) -> Result[dict]:
    stopped = await svc.task_manager.stop_task(request.session_id)
    if stopped:
        return Result[dict](success=True, data={"session_id": request.session_id, "stopped": True})
    return Result[dict](
        success=False,
        errorCode="SESSION_NOT_RUNNING",
        errorMessage=f"Session {request.session_id} is not currently running",
    )


# ========== Session Management ==========


@router.post(
    "/sessions/{session_id}/compact",
    response_model=Result[CompactSessionData],
    summary="Compact Chat Session",
    description="Compact chat session by summarizing conversation history",
)
async def compact_chat_session(
    session_id: Annotated[str, Path(description="Session ID to compact")],
    svc: ServiceDep,
    ctx: AppContextDep,
) -> Result[CompactSessionData]:
    return await svc.chat.compact_session(CompactSessionInput(session_id=session_id), user_id=ctx.user_id)


@router.get(
    "/sessions",
    response_model=Result[ChatSessionData],
    summary="List Chat Sessions",
    description=(
        "List chat sessions. Pass subagent_id to filter by agent "
        "(use 'chat' for the default chat agent, or any builtin/custom subagent id). "
        "Omit to return every session for the user."
    ),
)
async def list_sessions(
    svc: ServiceDep,
    ctx: AppContextDep,
    subagent_id: Optional[str] = Query(
        default=None,
        description="Filter by subagent id; 'chat' selects the default chat agent",
    ),
) -> Result[ChatSessionData]:
    return svc.chat.list_sessions(user_id=ctx.user_id, subagent_id=subagent_id)


@router.delete(
    "/sessions/{session_id}",
    response_model=Result[ChatSessionData],
    summary="Delete Chat Session",
    description="Delete a chat session by ID",
)
async def delete_session(
    session_id: Annotated[str, Path(description="Session ID to delete")],
    svc: ServiceDep,
    ctx: AppContextDep,
) -> Result[ChatSessionData]:
    return svc.chat.delete_session(session_id, user_id=ctx.user_id)


# ========== Chat History (GET /api/v1/history/chat?session_id=xxx) ==========


@router.get(
    "/history",
    response_model=Result[ChatHistoryData],
    summary="Get Chat History",
    description="Get full conversation messages for a chat session",
)
async def get_chat_history(
    svc: ServiceDep,
    ctx: AppContextDep,
    session_id: str = Query(..., description="Session ID to retrieve history for"),
) -> Result[ChatHistoryData]:
    return svc.chat.get_history(session_id, user_id=ctx.user_id)


# ========== User Interaction ==========


@router.post(
    "/user_interaction",
    response_model=Result[dict],
    summary="Submit User Interaction",
    description="Submit user's choice or input for an interactive dialog",
)
async def submit_user_interaction(
    request: UserInteractionInput,
    svc: ServiceDep,
) -> Result[dict]:
    task_manager = svc.task_manager
    task = task_manager.get_task(request.session_id)
    if task is None or task.node is None:
        return Result[dict](
            success=False,
            errorCode="SESSION_NOT_FOUND",
            errorMessage="No active task found for this session",
        )

    broker = task.node.interaction_broker
    if not broker:
        return Result[dict](
            success=False,
            errorCode="BROKER_NOT_FOUND",
            errorMessage="Interaction broker not found for this session",
        )

    # Validate: each answer must be non-empty
    if not request.input or any(len(ans) == 0 for ans in request.input):
        return Result[dict](
            success=False,
            errorCode="INVALID_INPUT",
            errorMessage="Each answer must contain at least one value",
        )

    success = await broker.submit(request.interaction_key, request.input)
    return Result[dict](
        success=success,
        data={"interaction_key": request.interaction_key, "submitted": success},
    )


# ========== Insert Message (mid-run user input) ==========


@router.post(
    "/insert",
    response_model=Result[InsertMessageData],
    summary="Insert user message into a running chat",
    description=(
        "Append a free-text user message to the agent's pending input queue. The message "
        "is delivered to the model before its next LLM turn within the same run via the "
        "OpenAI Agents SDK ``call_model_input_filter`` hook. If the run has already entered "
        "its final turn, the message will auto-continue the conversation in a follow-up run."
    ),
)
async def insert_message(
    request: InsertMessageInput,
    svc: ServiceDep,
) -> Result[InsertMessageData]:
    task_manager = svc.task_manager
    task = task_manager.get_task(request.session_id)
    if task is None or task.node is None:
        return Result[InsertMessageData](
            success=False,
            errorCode="SESSION_NOT_RUNNING",
            errorMessage="No active chat task for this session",
        )

    text = request.message.strip()
    if not text:
        return Result[InsertMessageData](
            success=False,
            errorCode="INVALID_INPUT",
            errorMessage="message must be non-empty after stripping whitespace",
        )

    queue = getattr(task.node, "pending_input_queue", None)
    if queue is None:
        return Result[InsertMessageData](
            success=False,
            errorCode="QUEUE_UNAVAILABLE",
            errorMessage="Pending input queue is not initialized for this session",
        )

    queue.push(text)
    return Result[InsertMessageData](
        success=True,
        data=InsertMessageData(session_id=request.session_id, queued_count=len(queue)),
    )


# ========== Tool Result ==========


@router.post(
    "/tool_result",
    response_model=Result[ToolResultData],
    summary="Submit Tool Execution Result",
    description="Receive tool execution result from frontend after filesystem operation",
)
async def submit_tool_result(
    request: ToolResultInput,
    svc: ServiceDep,
) -> Result[ToolResultData]:
    """Receive tool execution result from frontend."""
    task_manager = svc.task_manager
    task = task_manager.get_task(request.session_id) if request.session_id else None
    if not task or not task.node:
        return Result[ToolResultData](
            success=False,
            errorCode="TASK_NOT_FOUND",
            errorMessage="No active task found for this session",
        )

    await task.node.tool_channel.publish(request.call_tool_id, request.tool_result.model_dump())
    return Result[ToolResultData](
        success=True,
        data=ToolResultData(call_tool_id=request.call_tool_id, status="received"),
    )


# ========== Helpers ==========


def _sse_headers() -> dict:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "text/event-stream; charset=utf-8",
    }


# --------------------------------------------------------------------
# Hook integration helpers
# --------------------------------------------------------------------


async def _run_pre_chat_hook(
    hooks: Optional[ChatHooks],
    http_request: Request,
    request: StreamChatInput,
    user_id: Optional[str],
) -> Optional[ChatPreCheckOutcome]:
    """Invoke the pre-chat hook and translate exceptions into a denial.

    A hook that raises is treated as a server-error denial; we never
    fail-open here, because the host registered the hook precisely to
    gate access (e.g. credit balance).
    """
    if hooks is None:
        return None
    try:
        outcome = await hooks.pre_chat(http_request, request, user_id)
    except Exception as exc:  # pragma: no cover — defensive
        logger.error("pre_chat hook failed: %s", exc, exc_info=True)
        return ChatPreCheckOutcome(
            allow=False,
            error="Server error, please try again later.",
            error_type="PRE_CHAT_HOOK_ERROR",
        )
    if outcome is None:
        return ChatPreCheckOutcome(allow=True)
    return outcome


async def _emit_pre_check_denial(
    request: StreamChatInput,
    outcome: ChatPreCheckOutcome,
) -> AsyncGenerator[str, None]:
    """Emit a single SSE error event reflecting a pre-check denial."""
    error_payload = SSEErrorData(
        error=outcome.error or "Request denied",
        error_type=outcome.error_type or "PRE_CHAT_DENIED",
        session_id=request.session_id,
    )
    event = SSEEvent(id=1, event="error", data=error_payload, timestamp=now_utc_iso())
    yield f"id: {event.id}\nevent: {event.event}\ndata: {event.data.model_dump_json()}\n\n"


async def _stream_with_post_hook(
    upstream: AsyncGenerator[SSEEvent, None],
    *,
    http_request: Request,
    request: StreamChatInput,
    user_id: Optional[str],
    hooks: Optional[ChatHooks],
    pre_extra: dict,
) -> AsyncGenerator[str, None]:
    """Forward SSE events while capturing usage for the post-chat hook.

    The post hook is dispatched as a background task in ``finally`` so the
    response stream is never blocked waiting for billing to acknowledge.

    If the upstream generator raises before the stream completes, we log
    the error and emit a synthetic ``event: error`` to the client so the
    UI can surface a real reason instead of an opaque "network error".
    ``asyncio.CancelledError`` (client disconnect / shutdown) is treated
    as expected and skips the error event.
    """
    last_end_event: Optional[SSEEvent] = None
    captured_error: Optional[BaseException] = None
    was_cancelled: bool = False
    last_event_id: int = 0

    try:
        async for event in upstream:
            if event.event == "end":
                last_end_event = event
            if isinstance(event.id, int):
                last_event_id = event.id
            yield f"id: {event.id}\nevent: {event.event}\ndata: {event.data.model_dump_json()}\n\n"
    except asyncio.CancelledError:
        # Client disconnected or the server is shutting down. The HTTP
        # response is already gone — nothing to yield, nothing to log
        # as error. We also do NOT schedule the post-chat hook below,
        # because the usage payload is incomplete (no ``end`` event) and
        # would otherwise look like a free successful turn to callers
        # such as billing.
        was_cancelled = True
        raise
    except BaseException as exc:
        captured_error = exc
        logger.error(
            "stream_chat generator failed for session=%s user=%s subagent=%s: %s",
            request.session_id,
            user_id,
            request.subagent_id,
            exc,
            exc_info=True,
        )
        # Emit a synthetic SSE error event so the client renders a real
        # reason instead of an opaque "network error". Catch
        # ``BaseException`` here — including ``CancelledError`` raised
        # by ``yield`` when the peer has already disconnected — so the
        # outer bare ``raise`` below still re-propagates the original
        # ``exc`` instead of being masked.
        try:
            error_event = SSEEvent(
                id=last_event_id + 1,
                event="error",
                data=SSEErrorData(
                    error=str(exc) or type(exc).__name__,
                    error_type=type(exc).__name__,
                    session_id=request.session_id,
                ),
                timestamp=now_utc_iso(),
            )
            yield (f"id: {error_event.id}\nevent: {error_event.event}\ndata: {error_event.data.model_dump_json()}\n\n")
        except BaseException:  # pragma: no cover — defensive
            logger.warning(
                "Failed to emit terminal SSE error event (client likely disconnected)",
                exc_info=True,
            )
        raise
    finally:
        # Only schedule the post-chat hook when we have a meaningful
        # outcome to report: either the stream finished normally (an
        # ``end`` event was observed) or it failed with a real error
        # we can describe. Pure cancellation is skipped — the usage
        # payload would be empty and billing/audit callers would treat
        # the turn as a free success.
        should_schedule_post_hook = (
            hooks is not None and not was_cancelled and (last_end_event is not None or captured_error is not None)
        )
        if should_schedule_post_hook:
            usage_dict: dict = {}
            session_id_value: Optional[str] = request.session_id
            if last_end_event is not None:
                # SSEEndData fields cover requests / *_tokens / duration.
                usage_dict = last_end_event.data.model_dump()
                session_id_value = usage_dict.get("session_id") or session_id_value

            ctx = ChatPostUsageContext(
                user_id=user_id,
                session_id=session_id_value,
                model=request.model,
                usage=usage_dict,
                error=str(captured_error) if captured_error else None,
                pre_check_extra=dict(pre_extra),
            )

            try:
                asyncio.create_task(
                    _safe_post_chat(hooks, http_request, request, ctx),
                    name=f"chat-post-hook:{session_id_value or '-'}",
                )
            except Exception:  # pragma: no cover — defensive
                logger.error("Failed to schedule post_chat hook", exc_info=True)


async def _safe_post_chat(
    hooks: ChatHooks,
    http_request: Request,
    request: StreamChatInput,
    ctx: ChatPostUsageContext,
) -> None:
    """Run the post-chat hook and swallow exceptions.

    The hook is owned by the host and is responsible for its own retry
    policy; we only ensure that a misbehaving hook never crashes the
    background task in a way that propagates uncaught.
    """
    try:
        await hooks.post_chat(http_request, request, ctx)
    except Exception as exc:  # pragma: no cover — defensive
        logger.error("post_chat hook raised: %s", exc, exc_info=True)
