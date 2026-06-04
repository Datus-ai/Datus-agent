"""
Chat Task Manager — decouples the agentic loop into background asyncio.Tasks.

The agentic loop runs in a background Task, writing SSE events to a buffer.
SSE endpoints consume events from the buffer via ``consume_events``.
Disconnecting a client does **not** cancel the background computation;
the client can reconnect and resume from where it left off.
"""

import asyncio
import copy
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.api.models.cli_models import (
    IMessageContent,
    SSEDataType,
    SSEEndData,
    SSEErrorData,
    SSEEvent,
    SSEMessageData,
    SSEMessagePayload,
    SSEPingData,
    SSESessionData,
    SSEUsageData,
    StreamChatInput,
)
from datus.api.services.action_sse_converter import action_to_sse_event
from datus.cli.autocomplete import AtReferenceCompleter
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import Metric, ReferenceSql, TableSchema
from datus.tools.proxy.proxy_tool import apply_proxy_tools
from datus.utils.loggings import get_logger
from datus.utils.path_manager import set_current_path_manager
from datus.utils.time_utils import now_utc_iso
from datus.utils.trace_context import build_chat_trace_context, reset_trace_context, set_trace_context

logger = get_logger(__name__)

HEARTBEAT_INTERVAL = 10  # seconds


def is_thinking_only_content(content_items) -> bool:
    """Return True if all content items are thinking chunks (i.e. a delta message).

    Used by both the SSE coalescing logic and the bridge outbound conversion
    to avoid duplicating the detection heuristic.
    """
    return bool(content_items) and all(getattr(item, "type", "") == "thinking" for item in content_items)


def _is_thinking_delta(event: SSEEvent) -> bool:
    """Return True if *event* is a thinking delta (consecutive-mergeable)."""
    if event.event != "message":
        return False
    data = event.data
    if not isinstance(data, SSEMessageData):
        return False
    if data.type not in (SSEDataType.CREATE_MESSAGE, SSEDataType.APPEND_MESSAGE):
        return False
    return is_thinking_only_content(data.payload.content)


def _delta_message_id(event: SSEEvent) -> str:
    """Extract the message_id from a thinking-delta event.

    Callers must ensure *event* passes ``_is_thinking_delta`` first.
    """
    data = event.data
    if isinstance(data, SSEMessageData):
        return data.payload.message_id
    return ""


def _has_visible_content(event: SSEEvent) -> bool:
    if event.event != "message" or not isinstance(event.data, SSEMessageData):
        return False
    return any(bool(getattr(item, "payload", {}).get("content")) for item in event.data.payload.content)


def _assistant_content_fingerprint(event: SSEEvent) -> str:
    if event.event != "message" or not isinstance(event.data, SSEMessageData):
        return ""
    if event.data.payload.role != "assistant":
        return ""
    parts = []
    for item in event.data.payload.content:
        if item.type not in {"markdown", "thinking", "code"}:
            continue
        payload = getattr(item, "payload", {}) or {}
        content = payload.get("content")
        if content:
            parts.append(str(content).strip())
    return "\n".join(part for part in parts if part)


def _should_skip_duplicate_assistant_message(
    action,
    event: SSEEvent,
    seen_fingerprints: set[str],
) -> bool:
    if action.role != ActionRole.ASSISTANT or action.status != ActionStatus.SUCCESS:
        return False
    if action.action_type == "thinking_delta":
        return False
    if event.event != "message" or not isinstance(event.data, SSEMessageData):
        return False
    if event.data.type != SSEDataType.CREATE_MESSAGE:
        return False
    fingerprint = _assistant_content_fingerprint(event)
    return bool(fingerprint and fingerprint in seen_fingerprints)


def _remember_assistant_message(event: SSEEvent, seen_fingerprints: set[str]) -> None:
    fingerprint = _assistant_content_fingerprint(event)
    if fingerprint:
        seen_fingerprints.add(fingerprint)


def _should_include_final_response(action, assistant_response_sent: bool) -> bool:
    """Return True for top-level wrapper responses that should be rendered.

    Sub-agent actions are forwarded with ``depth > 0``. Their own
    ``*_response`` wrappers must stay inside the tool/sub-agent transcript and
    must not become the top-level assistant bubble.
    """
    return (
        action.role == ActionRole.ASSISTANT
        and action.status == ActionStatus.SUCCESS
        and getattr(action, "depth", 0) == 0
        and bool(action.action_type)
        and action.action_type.endswith("_response")
        and not assistant_response_sent
    )


def _is_visible_assistant_response(action, event: SSEEvent, *, tool_result_seen: bool) -> bool:
    """Return True when an action already emitted user-visible assistant text.

    Model providers do not agree on whether final text appears as ``response``,
    ``message`` or a completed thinking chunk. For web de-duping we care about
    the observable SSE message: after a tool result, any visible assistant text
    means the wrapper ``chat_response`` would duplicate it.
    """
    if action.role != ActionRole.ASSISTANT or action.status != ActionStatus.SUCCESS:
        return False
    if not action.action_type or action.action_type == "thinking_delta" or action.action_type.endswith("_response"):
        return False
    if not _has_visible_content(event):
        return False
    output = action.output if isinstance(action.output, dict) else {}
    return tool_result_seen or output.get("is_thinking") is not True


def _coalesce_deltas(events: list[SSEEvent]) -> list[SSEEvent]:
    """Merge consecutive thinking-delta events **for the same message** into single events.

    Non-delta events pass through unchanged and break any ongoing run of deltas.
    A change in ``message_id`` between adjacent deltas also breaks the run so
    that deltas from different logical messages are never merged together.
    """
    if not events:
        return []

    result: list[SSEEvent] = []
    run_start: int | None = None  # index of first delta in the current run
    run_msg_id: str = ""  # message_id of the current run

    for i, ev in enumerate(events):
        if _is_thinking_delta(ev):
            msg_id = _delta_message_id(ev)
            if run_start is None:
                run_start = i
                run_msg_id = msg_id
            elif msg_id != run_msg_id:
                # Different message — flush the current run and start a new one
                result.append(_merge_delta_run(events[run_start:i]))
                run_start = i
                run_msg_id = msg_id
        else:
            # Flush any accumulated delta run before emitting this non-delta
            if run_start is not None:
                result.append(_merge_delta_run(events[run_start:i]))
                run_start = None
            result.append(ev)

    # Flush trailing delta run
    if run_start is not None:
        result.append(_merge_delta_run(events[run_start:]))

    return result


def _merge_delta_run(run: list[SSEEvent]) -> SSEEvent:
    """Merge a non-empty run of thinking-delta events into a single event."""
    if len(run) == 1:
        return run[0]

    first = run[0]
    # Concatenate the text from content[0].payload["content"] of each event
    parts: list[str] = []
    for ev in run:
        data = ev.data
        if not isinstance(data, SSEMessageData):  # guaranteed by caller; guard for safety
            continue
        for item in data.payload.content:
            parts.append(item.payload.get("content", ""))

    merged_content_items = copy.deepcopy(first.data.payload.content)  # type: ignore[union-attr]
    # Replace the first item's text with the concatenated text
    if merged_content_items:
        merged_content_items[0].payload["content"] = "".join(parts)
        # Keep only one content item for the merged event
        merged_content_items = merged_content_items[:1]

    merged_payload = copy.deepcopy(first.data.payload)  # type: ignore[union-attr]
    merged_payload.content = merged_content_items
    merged_data = SSEMessageData(type=first.data.type, payload=merged_payload)  # type: ignore[union-attr]

    return SSEEvent(
        id=first.id,
        event=first.event,
        data=merged_data,
        timestamp=first.timestamp,
    )


def _fill_database_context(
    agent_config: Optional[AgentConfig],
    catalog: Optional[str] = None,
    database: Optional[str] = None,
    schema: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve API database context without changing the active datasource."""
    config = None
    if agent_config is not None:
        try:
            config = agent_config.current_db_config()
        except Exception:
            config = None

    def first_string(*values):
        for value in values:
            if isinstance(value, str) and value:
                return value
        return None

    return (
        first_string(catalog, getattr(config, "catalog", None)),
        first_string(database, getattr(config, "database", None)),
        first_string(schema, getattr(config, "schema", None)),
    )


class ChatTask:
    """Represents a single running agentic loop."""

    def __init__(self, session_id: str, asyncio_task: asyncio.Task):
        self.session_id = session_id
        self.asyncio_task = asyncio_task
        self.node: Optional[AgenticNode] = None
        self.events: list[SSEEvent] = []
        self.status: str = "running"  # running | completed | error | cancelled
        self.condition = asyncio.Condition()
        self.created_at = datetime.now()
        self.error: Optional[str] = None
        self.consumer_offset: int = 0


COMPLETED_TASK_TTL = 300  # seconds to keep completed tasks for resume


class ChatTaskManager:
    """Per-project manager for active chat tasks.

    Owned by DatusService — one instance per cached project.
    """

    def __init__(
        self,
        default_source: Optional[str] = None,
        default_interactive: bool = True,
        stream_thinking: bool = False,
    ) -> None:
        self._tasks: Dict[str, ChatTask] = {}
        self._completed_tasks: Dict[str, ChatTask] = {}
        self._default_source = default_source
        self._default_interactive = default_interactive
        self._stream_thinking = stream_thinking

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_chat(
        self,
        agent_config: AgentConfig,
        request: StreamChatInput,
        sub_agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ChatTask:
        """Create a background task for the agentic loop.
            :param sub_agent_id: builtin name or custom sub-agent DB ID
        Raises ``ValueError`` if a task is already running for the session.
        """
        # Clone config to avoid cross-request mutation of shared AgentConfig
        agent_config = copy.deepcopy(agent_config)
        # API surface has no interactive broker to confirm EXTERNAL file
        # access, so force filesystem strict mode — every node constructed
        # below reads this flag via AgenticNode._resolve_filesystem_strict().
        agent_config.filesystem_strict = True
        # Remote front-ends (vscode/web) own their own shell: the daemon must
        # not offer a server-side BashTool. ``project_root`` is intentionally
        # left untouched — web keeps its configured root, and the read-only
        # ``AgentConfig.project_root`` property already falls back to the
        # launch CWD when no root was supplied, so an empty project_root
        # naturally resolves to the current directory.
        effective_source = request.source or self._default_source
        if effective_source in ("vscode", "web"):
            agent_config.bash_tool_enabled = False
        # Stash the resolved source on the cloned config so downstream nodes
        # can adapt prompt-side hints to the front-end (e.g. vscode renders
        # the literal "." for the SQL files root because the IDE owns its own
        # workspace path).
        agent_config._client_source = effective_source
        # Per-request response language override. Empty / None keeps the
        # yaml-level ``agent.language`` default intact.
        if request.language:
            agent_config.language = request.language
        if request.model:
            provider, _, model_id = request.model.partition("/")
            if not model_id:
                raise ValueError(f"Invalid model format '{request.model}': expected 'provider/model_id'")
            if provider == "custom":
                agent_config.set_active_custom(model_id, persist=False)
            else:
                agent_config.set_active_provider_model(provider, model_id, persist=False)
        request.catalog, request.database, request.db_schema = _fill_database_context(
            agent_config,
            catalog=request.catalog,
            database=request.database,
            schema=request.db_schema,
        )
        agent_name = sub_agent_id or "chat"
        safe_name = agent_name.replace(" ", "_")
        session_id = request.session_id or f"{safe_name}_session_{str(uuid.uuid4())[:8]}"
        request.session_id = session_id

        if session_id in self._tasks:
            raise ValueError(f"A task is already running for session {session_id}")

        # Placeholder — asyncio_task set immediately after
        task = ChatTask(session_id=session_id, asyncio_task=None)  # type: ignore[arg-type]
        self._tasks[session_id] = task

        asyncio_task = asyncio.create_task(
            self._run_loop(task, agent_config, request, sub_agent_id=sub_agent_id, user_id=user_id)
        )
        task.asyncio_task = asyncio_task
        return task

    async def stop_task(self, session_id: str) -> bool:
        """Stop a running task by interrupting its node."""
        task = self._tasks.get(session_id)
        if not task:
            return False

        if task.node:
            try:
                task.node.interrupt_controller.interrupt()
                logger.info(f"Interrupted running task: {session_id}")
            except Exception as e:
                logger.error(f"Failed to interrupt task {session_id}: {e}")

        if task.asyncio_task and not task.asyncio_task.done():
            task.asyncio_task.cancel()
            logger.info(f"Cancelled asyncio task: {session_id}")
            return True

        return False

    def has_active_tasks(self) -> bool:
        """Return True if any task is still running."""
        return any(t.status == "running" for t in self._tasks.values())

    def get_task(self, session_id: str) -> Optional[ChatTask]:
        return self._tasks.get(session_id) or self._completed_tasks.get(session_id)

    async def consume_events(self, task: ChatTask, start_from: Optional[int] = None) -> AsyncGenerator[SSEEvent, None]:
        """Yield events from *task*'s buffer.

        If *start_from* is ``None``, resume from the last recorded
        ``consumer_offset`` — but back up by one event so the client
        can safely re-process the last event it may not have fully handled.
        """
        if start_from is not None:
            cursor = start_from
        else:
            cursor = max(task.consumer_offset - 1, 0)

        while True:
            ping_event = None
            async with task.condition:
                while cursor >= len(task.events) and task.status == "running":
                    try:
                        await asyncio.wait_for(task.condition.wait(), timeout=HEARTBEAT_INTERVAL)
                    except asyncio.TimeoutError:
                        if cursor >= len(task.events) and task.status == "running":
                            ping_event = SSEEvent(
                                id=-1,
                                event="ping",
                                data=SSEPingData(),
                                timestamp=now_utc_iso(),
                            )
                            break  # exit inner loop so ping can be yielded
                new_events = task.events[cursor:]
                is_done = task.status != "running"

            # Yield outside the lock to avoid blocking producers
            if ping_event is not None:
                yield ping_event

            coalesced = _coalesce_deltas(new_events)
            for event in coalesced:
                yield event
            cursor += len(new_events)
            task.consumer_offset = cursor

            if is_done and cursor >= len(task.events):
                break

    async def wait_all_tasks(self) -> None:
        """Wait for all running tasks to finish without cancelling them."""
        pending = [t.asyncio_task for t in self._tasks.values() if t.asyncio_task and not t.asyncio_task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def shutdown(self) -> None:
        """Cancel every running task (called at application shutdown)."""
        for task in list(self._tasks.values()):
            if task.asyncio_task and not task.asyncio_task.done():
                task.asyncio_task.cancel()
        pending = [t.asyncio_task for t in self._tasks.values() if t.asyncio_task and not t.asyncio_task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._completed_tasks.clear()

    # ------------------------------------------------------------------
    # Background loop (full agentic loop implementation)
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        task: ChatTask,
        agent_config: AgentConfig,
        request: StreamChatInput,
        sub_agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Execute the full agentic loop, pushing SSE events to the task buffer."""
        session_id = task.session_id
        event_id = 0
        trace_token = None

        # Pin the path manager into this task's context. Required when the caller
        # dispatched us from a thread that never inherited AgentConfig's ContextVar
        # (e.g. gateway bridge dispatching from an IM SDK worker thread via
        # ``asyncio.run_coroutine_threadsafe``); otherwise downstream stores fall
        # back to ``get_path_manager()`` and get an empty project_name.
        set_current_path_manager(agent_config.path_manager)

        try:
            start_time = datetime.now()

            # 1. Create node.
            #    Runs in thread pool because setup_tools() triggers synchronous
            #    operations (psycopg ConnectionPool creation, PG DDL for table
            #    creation via get_storage()) that would freeze the event loop.
            interactive_enabled = request.interactive if request.interactive is not None else self._default_interactive

            def _init_node():
                # Feedback runs triggered with a source_session_id pre-copy the
                # source conversation into a fresh feedback session file BEFORE
                # node construction. The node then opens that cloned id directly
                # — no post-construction mutation needed.
                feedback_session_id: Optional[str] = None
                if sub_agent_id == "feedback" and request.source_session_id:
                    from datus.models.session_manager import SessionManager
                    from datus.utils.path_manager import get_path_manager

                    base_dir = getattr(agent_config, "session_dir", None) or str(
                        get_path_manager(agent_config=agent_config).sessions_dir
                    )
                    sm = SessionManager(session_dir=base_dir, scope=user_id)
                    feedback_session_id = sm.copy_session(request.source_session_id, "feedback")

                return self._create_node(
                    agent_config,
                    subagent_id=sub_agent_id,
                    node_id=session_id,
                    user_id=user_id,
                    interactive=interactive_enabled,
                    session_id=feedback_session_id,
                )

            node = await asyncio.to_thread(_init_node)
            task.node = node
            trace_token = set_trace_context(
                build_chat_trace_context(
                    session_id=session_id,
                    llm_session_id=node.session_id,
                    node_name=node.get_node_name() if hasattr(node, "get_node_name") else None,
                    subagent_id=sub_agent_id,
                    user_id=user_id,
                    datasource=agent_config.current_datasource,
                    source_session_id=request.source_session_id,
                    source=request.source or self._default_source,
                    model=request.model,
                    agent_home=agent_config.home,
                )
            )

            # Per-request permission profile override. We deliberately do
            # NOT mutate ``agent_config.active_profile_name`` here because
            # the AgentConfig instance is shared across concurrent SaaS
            # users; rewriting it on one request would leak the new profile
            # to every other in-flight or future request. Instead we
            # switch the freshly created node's PermissionManager in
            # place — it is scoped to this request only.
            self._apply_permission_mode_override(node, agent_config, request.permission_mode)

            await self._push_event(
                task,
                SSEEvent(
                    id=event_id,
                    event="session",
                    data=SSESessionData(
                        session_id=session_id,
                        llm_session_id=node.session_id,
                    ),
                    timestamp=now_utc_iso(),
                ),
            )
            event_id += 1
            event_id = await self._push_degraded_capability_warnings(task, node, event_id)

            # 3. Resolve @-references
            at_tables, at_metrics, at_sqls = self._resolve_at_context(
                agent_config, request.table_paths, request.metric_paths, request.sql_paths
            )

            # 4. Build typed input and assign to node
            node_input = self._create_node_input(
                user_message=request.message,
                current_node=node,
                at_tables=at_tables,
                at_metrics=at_metrics,
                at_sqls=at_sqls,
                catalog=request.catalog,
                database=request.database,
                db_schema=request.db_schema,
                plan_mode=request.plan_mode or False,
                source_session_id=request.source_session_id,
            )
            node.input = node_input

            # 5. Replace filesystem tools with proxy if applicable.
            # ``apply_proxy_tools`` consults ``_FS_DEPENDENT_NODES`` and the
            # node's ``tool_registry`` to leave filesystem tools un-proxied
            # for nodes that author server-side artifacts (e.g.
            # ``gen_visual_report`` writing ``render/*.jsx``). No isinstance
            # guard is needed here.
            effective_source = request.source or self._default_source
            if effective_source == "vscode":
                apply_proxy_tools(node, ["filesystem_tools.*"])
            elif effective_source == "web":
                apply_proxy_tools(node, ["write_file", "edit_file"])
            elif effective_source:
                logger.warning("Unsupported source '%s'; skipping proxy shortcut", effective_source)

            # 6. Execute streaming
            action_history = ActionHistoryManager()
            action_count = 0
            seen_delta_action_ids: set[str] = set()
            assistant_response_sent = False
            tool_result_seen = False
            seen_assistant_message_fingerprints: set[str] = set()

            async for action in node.execute_stream_with_interactions(action_history):
                action_count += 1

                # Convert action to SSE
                # Per-request stream_response overrides the server-level --stream flag
                effective_stream = (
                    request.stream_response if request.stream_response is not None else self._stream_thinking
                )

                is_first_delta = True
                if action.action_type == "thinking_delta":
                    is_first_delta = action.action_id not in seen_delta_action_ids
                    seen_delta_action_ids.add(action.action_id)

                # finalize_progress actions reuse the same id across stages
                # so the SSE wire emits CREATE then UPDATE_MESSAGE; we mark
                # everything past the first emission as an update.
                is_finalize_progress_update = False
                if action.action_type == "finalize_progress":
                    is_finalize_progress_update = action.action_id in seen_delta_action_ids
                    seen_delta_action_ids.add(action.action_id)

                is_update = is_finalize_progress_update or (
                    effective_stream
                    and action.action_type == "response"
                    and isinstance(action.output, dict)
                    and action.action_id in seen_delta_action_ids
                )

                sse = action_to_sse_event(
                    action,
                    event_id,
                    action.action_id,
                    stream_thinking=effective_stream,
                    is_first_delta=is_first_delta,
                    is_update=bool(is_update),
                    include_final_response=_should_include_final_response(action, assistant_response_sent),
                )
                if sse:
                    # Per-LLM-call usage event: the converter has no access
                    # to the service-level session ids, so we stamp them
                    # here before fan-out. Skip the assistant-message dedup
                    # path entirely since usage carries no rendered text.
                    if sse.event == "usage" and isinstance(sse.data, SSEUsageData):
                        sse.data.session_id = session_id
                        # Only main-agent usage (depth==0) belongs to this
                        # node's LLM session. Sub-agent usage (depth>0) keeps the
                        # sub-agent session id stamped by the converter so the
                        # consumer can attribute it to the right session instead
                        # of mislabelling it as the parent's.
                        if sse.data.depth == 0:
                            sse.data.llm_session_id = node.session_id
                        await self._push_event(task, sse)
                        event_id += 1
                        continue
                    if _should_skip_duplicate_assistant_message(
                        action,
                        sse,
                        seen_assistant_message_fingerprints,
                    ):
                        continue
                    await self._push_event(task, sse)
                    event_id += 1
                    _remember_assistant_message(sse, seen_assistant_message_fingerprints)
                    if _is_visible_assistant_response(action, sse, tool_result_seen=tool_result_seen):
                        assistant_response_sent = True
                    if action.role == ActionRole.TOOL and action.status != ActionStatus.PROCESSING:
                        tool_result_seen = True

            # 7. End event
            token_kwargs: dict = {}
            try:
                turn_usage = await node.get_last_turn_usage()
                if turn_usage:
                    token_kwargs = {
                        "requests": turn_usage.requests,
                        "input_tokens": turn_usage.input_tokens,
                        "output_tokens": turn_usage.output_tokens,
                        "total_tokens": turn_usage.total_tokens,
                        "cached_tokens": turn_usage.cached_tokens,
                        "session_total_tokens": turn_usage.session_total_tokens,
                        "context_length": turn_usage.context_length,
                    }
            except Exception:
                logger.debug("Failed to extract turn token usage for end event", exc_info=True)

            await self._push_event(
                task,
                SSEEvent(
                    id=event_id,
                    event="end",
                    data=SSEEndData(
                        session_id=session_id,
                        llm_session_id=node.session_id,
                        total_events=event_id,
                        action_count=action_count,
                        duration=(datetime.now() - start_time).total_seconds(),
                        **token_kwargs,
                    ),
                    timestamp=now_utc_iso(),
                ),
            )
            event_id += 1

            task.status = "completed"

        except asyncio.CancelledError:
            task.status = "cancelled"

        except Exception as e:
            logger.error(f"Chat task error for session {session_id}: {e}")
            task.status = "error"
            task.error = str(e)
            await self._push_event(
                task,
                SSEEvent(
                    id=event_id,
                    event="error",
                    data=SSEErrorData(
                        error=str(e),
                        error_type=type(e).__name__,
                        session_id=session_id,
                        llm_session_id=task.node.session_id if task.node else None,
                    ),
                    timestamp=now_utc_iso(),
                ),
            )
            event_id += 1

        finally:
            if trace_token is not None:
                reset_trace_context(trace_token)
            async with task.condition:
                task.condition.notify_all()
            self._tasks.pop(session_id, None)
            # Keep completed task for resume within TTL
            self._completed_tasks[session_id] = task
            self._purge_expired_completed()

    async def _push_event(self, task: ChatTask, event: SSEEvent) -> None:
        """Append an event to the task buffer and notify consumers."""
        logger.debug(f"Pushing event: {event}")
        async with task.condition:
            task.events.append(event)
            task.condition.notify_all()

    def _purge_expired_completed(self) -> None:
        """Remove completed tasks older than COMPLETED_TASK_TTL."""
        now = datetime.now()
        expired = [
            sid for sid, t in self._completed_tasks.items() if (now - t.created_at).total_seconds() > COMPLETED_TASK_TTL
        ]
        for sid in expired:
            self._completed_tasks.pop(sid, None)

    # ------------------------------------------------------------------
    # Node factory
    # ------------------------------------------------------------------

    def _create_node(
        self,
        agent_config: AgentConfig,
        subagent_id: Optional[str],
        node_id: str,
        user_id: Optional[str] = None,
        interactive: bool = True,
        session_id: Optional[str] = None,
    ) -> AgenticNode:
        """Create a fresh AgenticNode based on subagent_id (builtin name or custom DB ID).

        Delegates dispatch to :func:`datus.agent.node.node_factory.create_interactive_node`
        so the API path matches the CLI exactly: every built-in sub_agent is wired to
        its dedicated AgenticNode subclass, and custom sub_agents honour their
        ``node_class`` field (``gen_report`` / ``gen_table`` / ``gen_dashboard`` /
        ``scheduler`` / ``gen_skill`` / ``explore``) instead of always falling back
        to ``GenSQLAgenticNode``.

        ``user_id`` is propagated as the node ``scope`` so that session files
        are isolated per user under ``{session_dir}/{user_id}/``. ``session_id``
        becomes the on-disk session identifier (defaults to ``node_id``); the
        feedback flow passes a pre-copied id so the new node opens the cloned
        session file directly instead of mutating ``node.session_id`` later.
        """
        from datus.agent.node.node_factory import create_interactive_node

        execution_mode: Literal["interactive", "workflow"] = "interactive" if interactive else "workflow"

        # ``agentic_nodes`` is keyed by sanitized node_name; the API receives the
        # custom sub_agent's UUID under the "id" field. Translate UUID -> name so
        # the factory's ``_resolve_node_class_type`` can look up node_class and
        # downstream tools can resolve scoped_context via sub_agent_config().
        node_name = subagent_id
        if subagent_id:
            for key, entry in (agent_config.agentic_nodes or {}).items():
                entry_id = entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", None)
                if entry_id == subagent_id:
                    node_name = key
                    break

        return create_interactive_node(
            subagent_name=node_name,
            agent_config=agent_config,
            scope=user_id,
            execution_mode=execution_mode,
            node_id=node_id,
            session_id=session_id if session_id is not None else node_id,
        )

    # ------------------------------------------------------------------
    # Per-request permission profile override
    # ------------------------------------------------------------------

    def _apply_permission_mode_override(
        self,
        node: AgenticNode,
        agent_config: AgentConfig,
        permission_mode: Optional[str],
    ) -> None:
        """Apply a per-request permission profile to the freshly created node.

        Switches ``node.permission_manager`` to ``permission_mode`` without
        touching ``agent_config.active_profile_name`` — the AgentConfig is
        shared by every concurrent request in the SaaS deployment, so
        mutating it would leak the override across users. The CLI's
        ``/profile`` flow can still mutate the global field because it
        owns the process exclusively; this API path cannot.

        No-ops when ``permission_mode`` is falsy, the node has no
        ``permission_manager`` (e.g. workflow nodes that skip the skill
        setup), or the requested profile already matches the active one.
        Failure handling is split deliberately:

        * Building ``user_overrides`` from ``agent.yml`` fails closed —
          raises so the outer ``_run_loop`` aborts the turn and emits an
          SSE error. Silently dropping malformed user rules would apply
          the bare profile base, which can be **broader** than the
          operator-configured posture (e.g. yaml had an explicit DENY
          we'd lose), so the safe move is to refuse the switch loudly.
        * ``switch_profile`` failures (unknown profile, malformed merge
          result) are logged and swallowed because at that point the
          node still has its original, server-default profile installed.
        """
        if not permission_mode:
            return
        permission_manager = getattr(node, "permission_manager", None)
        if permission_manager is None:
            return
        if getattr(permission_manager, "active_profile", None) == permission_mode:
            return

        from datus.tools.permission.profiles import build_user_overrides

        raw_permissions = getattr(agent_config, "_raw_permissions", {}) or {}
        raw_user = {k: v for k, v in raw_permissions.items() if k != "profile"}
        try:
            user_overrides = build_user_overrides(permission_mode, raw_user)
        except Exception as exc:
            logger.error(
                "Cannot build user overrides for permission_mode=%r from agent.yml: %s; "
                "refusing to switch profile to avoid broadening permissions beyond the "
                "operator-configured rules",
                permission_mode,
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to apply permission_mode={permission_mode!r}: agent.yml permissions.rules is malformed ({exc})"
            ) from exc

        try:
            permission_manager.switch_profile(permission_mode, user_overrides=user_overrides)
        except Exception as e:
            logger.error(
                "Failed to switch permission profile to %r for session=%s: %s",
                permission_mode,
                getattr(node, "session_id", None),
                e,
            )
            return

        logger.info(
            "Applied per-request permission profile %r for session=%s",
            permission_mode,
            getattr(node, "session_id", None),
        )

    # ------------------------------------------------------------------
    # Node input factory
    # ------------------------------------------------------------------

    def _create_node_input(
        self,
        user_message: str,
        current_node: AgenticNode,
        at_tables: List[TableSchema],
        at_metrics: List[Metric],
        at_sqls: List[ReferenceSql],
        catalog: Optional[str] = None,
        database: Optional[str] = None,
        db_schema: Optional[str] = None,
        plan_mode: bool = False,
        source_session_id: Optional[str] = None,
    ):
        """Create node input based on node type.

        Delegates to :func:`datus.agent.node.node_factory.create_node_input` so
        the API path covers every AgenticNode subclass the CLI knows about
        (GenReport / Explore / SkillCreator / GenTable / GenJob in addition to
        the GenSQL / Semantic / SqlSummary / Feedback / Chat branches).
        """
        from datus.agent.node.node_factory import create_node_input

        node_agent_config = getattr(current_node, "agent_config", None)
        if not isinstance(node_agent_config, AgentConfig):
            node_agent_config = None
        catalog, database, db_schema = _fill_database_context(
            node_agent_config,
            catalog=catalog,
            database=database,
            schema=db_schema,
        )

        return create_node_input(
            user_message=user_message,
            node=current_node,
            catalog=catalog,
            database=database,
            db_schema=db_schema,
            at_tables=at_tables,
            at_metrics=at_metrics,
            at_sqls=at_sqls,
            prompt_language="en",
            plan_mode=plan_mode,
            source_session_id=source_session_id,
        )

    # ------------------------------------------------------------------
    # @ reference resolution
    # ------------------------------------------------------------------

    async def _push_degraded_capability_warnings(self, task: ChatTask, node: AgenticNode, event_id: int) -> int:
        degraded = getattr(node, "degraded_capabilities", {}) or {}
        context_warning = degraded.get("context_search_tools")
        if not context_warning:
            return event_id

        await self._push_event(
            task,
            SSEEvent(
                id=event_id,
                event="message",
                data=SSEMessageData(
                    type=SSEDataType.CREATE_MESSAGE,
                    payload=SSEMessagePayload(
                        message_id=f"context-degraded-{uuid.uuid4().hex[:8]}",
                        role="assistant",
                        content=[
                            IMessageContent(
                                type="markdown",
                                payload={"content": context_warning},
                            )
                        ],
                    ),
                ),
                timestamp=now_utc_iso(),
            ),
        )
        return event_id + 1

    def _resolve_at_context(
        self,
        agent_config: AgentConfig,
        table_paths: Optional[List[str]],
        metric_paths: Optional[List[str]],
        sql_paths: Optional[List[str]],
    ) -> tuple[List[TableSchema], List[Metric], List[ReferenceSql]]:
        """Resolve @-reference paths to typed objects using a fresh completer."""
        try:
            completer = AtReferenceCompleter(agent_config)
            completer.reload_data()
        except Exception as exc:
            logger.warning("Failed to resolve @ references; continuing without context references: %s", exc)
            return [], [], []

        tables: List[TableSchema] = []
        for path in table_paths or []:
            try:
                entry = completer.table_completer.flatten_data.get(path)
                if entry:
                    tables.append(TableSchema.from_dict(entry))
            except Exception as e:
                logger.warning(f"Failed to resolve table path '{path}': {e}")

        metrics: List[Metric] = []
        for path in metric_paths or []:
            try:
                entry = completer.metric_completer.flatten_data.get(path)
                if entry:
                    metrics.append(Metric.from_dict(entry))
            except Exception as e:
                logger.warning(f"Failed to resolve metric path '{path}': {e}")

        sqls: List[ReferenceSql] = []
        for path in sql_paths or []:
            try:
                entry = completer.sql_completer.flatten_data.get(path)
                if entry:
                    sqls.append(ReferenceSql.from_dict(entry))
            except Exception as e:
                logger.warning(f"Failed to resolve sql path '{path}': {e}")

        return tables, metrics, sqls
