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
from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.api.models.cli_models import (
    SSEDataType,
    SSEEndData,
    SSEErrorData,
    SSEEvent,
    SSEMessageData,
    SSEPingData,
    SSESessionData,
    StreamChatInput,
)
from datus.api.services.action_sse_converter import action_to_sse_event
from datus.cli.autocomplete import AtReferenceCompleter
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager
from datus.schemas.node_models import Metric, ReferenceSql, TableSchema
from datus.tools.proxy.proxy_tool import apply_proxy_tools
from datus.utils.loggings import get_logger
from datus.utils.path_manager import set_current_path_manager

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
    agent_config: AgentConfig,
    catalog: Optional[str] = None,  # noqa: ARG001 — reserved for future use
    database: Optional[str] = None,
    schema: Optional[str] = None,  # noqa: ARG001 — reserved for future use
) -> None:
    """Set agent_config's current_namespace and current_database based on request context.

    If database is provided, searches for it in agent_config.namespaces and updates both
    current_namespace and current_database. Otherwise, falls back to treating database
    as a namespace name if it exists in agent_config.namespaces.
    """
    if not database:
        return
    for ns_name, ns_dbs in agent_config.namespaces.items():
        if database in ns_dbs:
            if agent_config.current_namespace != ns_name:
                agent_config.current_namespace = ns_name
            agent_config.current_database = database
            return
    # Fallback: database itself is a namespace name
    if database in agent_config.namespaces:
        agent_config.current_namespace = database


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
        _fill_database_context(
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
                                timestamp=datetime.now().isoformat() + "Z",
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

        # Pin the path manager into this task's context. Required when the caller
        # dispatched us from a thread that never inherited AgentConfig's ContextVar
        # (e.g. claw bridge dispatching from an IM SDK worker thread via
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
                n = self._create_node(
                    agent_config,
                    subagent_id=sub_agent_id,
                    session_id=session_id,
                    user_id=user_id,
                    interactive=interactive_enabled,
                )
                # Feedback runs triggered with a source_session_id must start with
                # no session_id so FeedbackAgenticNode.execute_stream copies the
                # source session into a fresh feedback_session_ id.
                if sub_agent_id == "feedback" and request.source_session_id:
                    n.session_id = None
                else:
                    n.session_id = session_id
                return n

            node = await asyncio.to_thread(_init_node)
            task.node = node

            await self._push_event(
                task,
                SSEEvent(
                    id=event_id,
                    event="session",
                    data=SSESessionData(
                        session_id=session_id,
                        llm_session_id=node.session_id,
                    ),
                    timestamp=datetime.now().isoformat() + "Z",
                ),
            )
            event_id += 1

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

            # 5. Replace filesystem tools with proxy if applicable
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

                is_update = (
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
                )
                if sse:
                    await self._push_event(task, sse)
                    event_id += 1

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
                    timestamp=datetime.now().isoformat() + "Z",
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
                    timestamp=datetime.now().isoformat() + "Z",
                ),
            )
            event_id += 1

        finally:
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
        session_id: str,
        user_id: Optional[str] = None,
        interactive: bool = True,
    ) -> AgenticNode:
        """Create a fresh AgenticNode based on subagent_id (builtin name or custom DB ID).

        ``user_id`` is propagated as the node ``scope`` so that session files
        are isolated per user under ``{session_dir}/{user_id}/``.
        """
        execution_mode: Literal["interactive", "workflow"] = "interactive" if interactive else "workflow"
        if subagent_id:
            if subagent_id == "gen_semantic_model":
                from datus.agent.node.gen_semantic_model_agentic_node import (
                    GenSemanticModelAgenticNode,
                )

                return GenSemanticModelAgenticNode(
                    agent_config=agent_config,
                    execution_mode=execution_mode,
                    scope=user_id,
                )
            elif subagent_id == "gen_metrics":
                from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

                return GenMetricsAgenticNode(
                    agent_config=agent_config,
                    execution_mode=execution_mode,
                    scope=user_id,
                )
            elif subagent_id == "gen_sql_summary":
                from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

                return SqlSummaryAgenticNode(
                    node_name=subagent_id,
                    agent_config=agent_config,
                    execution_mode=execution_mode,
                    scope=user_id,
                )
            elif subagent_id == "gen_ext_knowledge":
                from datus.agent.node.gen_ext_knowledge_agentic_node import (
                    GenExtKnowledgeAgenticNode,
                )

                return GenExtKnowledgeAgenticNode(
                    node_name=subagent_id,
                    agent_config=agent_config,
                    execution_mode=execution_mode,
                    scope=user_id,
                )
            elif subagent_id == "feedback":
                from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

                return FeedbackAgenticNode(
                    agent_config=agent_config,
                    execution_mode=execution_mode,
                    scope=user_id,
                )
            else:
                # Custom sub_agent: agentic_nodes is keyed by sanitized node_name
                # (not the UUID subagent_id). Each entry carries its original
                # sub_agent id under the "id" field — use it to resolve the key
                # so downstream tools can look up scoped_context via
                # sub_agent_config().
                node_name = subagent_id
                for key, entry in (agent_config.agentic_nodes or {}).items():
                    if isinstance(entry, dict) and entry.get("id") == subagent_id:
                        node_name = key
                        break
                return GenSQLAgenticNode(
                    node_id=session_id,
                    description=f"SQL generation node for {node_name}",
                    node_type="gensql",
                    input_data=None,
                    agent_config=agent_config,
                    tools=None,
                    node_name=node_name,
                    scope=user_id,
                    execution_mode=execution_mode,
                )
        else:
            return ChatAgenticNode(
                node_id=session_id,
                description="Chat node for backend API",
                node_type="chat",
                input_data=None,
                agent_config=agent_config,
                tools=None,
                scope=user_id,
                execution_mode=execution_mode,
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
        """Create node input based on node type."""
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

        if isinstance(current_node, FeedbackAgenticNode):
            from datus.schemas.feedback_agentic_node_models import FeedbackNodeInput

            return FeedbackNodeInput(
                user_message=user_message,
                database=database,
                source_session_id=source_session_id,
            )

        if isinstance(current_node, (GenSemanticModelAgenticNode, GenMetricsAgenticNode)):
            from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

            return SemanticNodeInput(
                user_message=user_message,
                catalog=catalog,
                database=database,
                db_schema=db_schema,
                prompt_language="en",
            )
        elif isinstance(current_node, SqlSummaryAgenticNode):
            from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

            return SqlSummaryNodeInput(
                user_message=user_message,
                catalog=catalog,
                database=database,
                db_schema=db_schema,
                prompt_language="en",
            )
        elif isinstance(current_node, GenExtKnowledgeAgenticNode):
            from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput

            return ExtKnowledgeNodeInput(
                user_message=user_message,
                prompt_language="en",
            )
        elif isinstance(current_node, GenSQLAgenticNode):
            from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

            return GenSQLNodeInput(
                user_message=user_message,
                catalog=catalog,
                database=database,
                db_schema=db_schema,
                schemas=at_tables,
                metrics=at_metrics,
                reference_sql=at_sqls,
                prompt_language="en",
                plan_mode=plan_mode,
            )
        else:
            from datus.schemas.chat_agentic_node_models import ChatNodeInput

            return ChatNodeInput(
                user_message=user_message,
                catalog=catalog,
                database=database,
                db_schema=db_schema,
                schemas=at_tables,
                metrics=at_metrics,
                reference_sql=at_sqls,
                plan_mode=plan_mode,
            )

    # ------------------------------------------------------------------
    # @ reference resolution
    # ------------------------------------------------------------------

    def _resolve_at_context(
        self,
        agent_config: AgentConfig,
        table_paths: Optional[List[str]],
        metric_paths: Optional[List[str]],
        sql_paths: Optional[List[str]],
    ) -> tuple[List[TableSchema], List[Metric], List[ReferenceSql]]:
        """Resolve @-reference paths to typed objects using a fresh completer."""
        completer = AtReferenceCompleter(agent_config)
        completer.reload_data()

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
