"""Tests for datus.api.services.chat_task_manager — background task management."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from datus.api.models.cli_models import (
    IMessageContent,
    SSEDataType,
    SSEEvent,
    SSEMessageData,
    SSEMessagePayload,
    SSEPingData,
)
from datus.api.services.chat_task_manager import (
    ChatTask,
    ChatTaskManager,
    _coalesce_deltas,
    _fill_database_context,
    _is_thinking_delta,
)


class TestFillDatabaseContext:
    """Tests for _fill_database_context — namespace/database resolution."""

    def test_no_database_is_noop(self, real_agent_config):
        """No database parameter leaves config unchanged."""
        original_ns = real_agent_config.current_namespace
        _fill_database_context(real_agent_config, database=None)
        assert real_agent_config.current_namespace == original_ns

    def test_empty_database_is_noop(self, real_agent_config):
        """Empty string database leaves config unchanged."""
        original_ns = real_agent_config.current_namespace
        _fill_database_context(real_agent_config, database="")
        assert real_agent_config.current_namespace == original_ns

    def test_known_database_updates_namespace_and_db(self, real_agent_config):
        """Known database in namespaces updates current_namespace and current_database."""
        # real_agent_config has "california_schools" in services.datasources
        # After the namespace→services.datasources refactor, each DB is its own namespace key
        _fill_database_context(real_agent_config, database="california_schools")
        assert real_agent_config.current_namespace == "california_schools"
        assert real_agent_config.current_database == "california_schools"

    def test_database_as_namespace_name(self, real_agent_config):
        """Database matching a namespace name falls back to namespace lookup."""
        # After refactor, namespace keys equal database names
        _fill_database_context(real_agent_config, database="california_schools")
        assert real_agent_config.current_namespace == "california_schools"

    def test_unknown_database_leaves_unchanged(self, real_agent_config):
        """Unknown database leaves config unchanged."""
        original_ns = real_agent_config.current_namespace
        _fill_database_context(real_agent_config, database="nonexistent_db")
        assert real_agent_config.current_namespace == original_ns


class TestChatTaskInit:
    """Tests for ChatTask initialization."""

    def test_initial_state(self):
        """ChatTask has correct initial state."""
        mock_task = MagicMock(spec=asyncio.Task)
        task = ChatTask(session_id="sess-1", asyncio_task=mock_task)

        assert task.session_id == "sess-1"
        assert task.asyncio_task is mock_task
        assert task.node is None
        assert task.events == []
        assert task.status == "running"
        assert task.error is None
        assert task.consumer_offset == 0
        assert isinstance(task.created_at, datetime)


class TestChatTaskManagerInit:
    """Tests for ChatTaskManager initialization."""

    def test_starts_empty(self):
        """Manager starts with no tasks."""
        manager = ChatTaskManager()
        assert manager._tasks == {}

    def test_default_source_and_interactive_defaults(self):
        """Defaults: source=None, interactive=True."""
        manager = ChatTaskManager()
        assert manager._default_source is None
        assert manager._default_interactive is True

    def test_default_source_and_interactive_stored(self):
        """Constructor stores explicit defaults."""
        manager = ChatTaskManager(default_source="web", default_interactive=False)
        assert manager._default_source == "web"
        assert manager._default_interactive is False


class TestChatTaskManagerCreateNodeInteractive:
    """Verify _create_node forwards interactive flag to ChatAgenticNode."""

    def test_create_node_passes_interactive_false(self, real_agent_config, mock_llm_create):
        """interactive=False reaches ChatAgenticNode and disables ask_user_tool."""
        manager = ChatTaskManager(default_interactive=False)
        node = manager._create_node(
            real_agent_config,
            subagent_id=None,
            session_id="sess-1",
            user_id=None,
            interactive=False,
        )
        assert node.execution_mode == "workflow"
        assert node.ask_user_tool is None

    def test_create_node_passes_interactive_true(self, real_agent_config, mock_llm_create):
        """interactive=True retains ask_user_tool setup."""
        manager = ChatTaskManager()
        node = manager._create_node(
            real_agent_config,
            subagent_id=None,
            session_id="sess-2",
            user_id=None,
            interactive=True,
        )
        assert node.execution_mode == "interactive"

    def test_has_active_tasks_returns_false_when_empty(self):
        """has_active_tasks is False when no tasks exist."""
        manager = ChatTaskManager()
        assert manager.has_active_tasks() is False

    def test_get_task_returns_none_for_missing(self):
        """get_task returns None for non-existent session."""
        manager = ChatTaskManager()
        assert manager.get_task("nonexistent") is None


class TestChatTaskManagerBehavior:
    """Tests for ChatTaskManager task tracking."""

    def test_has_active_tasks_true_when_running(self):
        """has_active_tasks returns True when a task has running status."""
        manager = ChatTaskManager()
        task = ChatTask(session_id="s1", asyncio_task=MagicMock())
        task.status = "running"
        manager._tasks["s1"] = task
        assert manager.has_active_tasks() is True

    def test_has_active_tasks_false_when_completed(self):
        """has_active_tasks returns False when all tasks are completed."""
        manager = ChatTaskManager()
        task = ChatTask(session_id="s1", asyncio_task=MagicMock())
        task.status = "completed"
        manager._tasks["s1"] = task
        assert manager.has_active_tasks() is False

    def test_get_task_returns_existing(self):
        """get_task returns the task for an existing session."""
        manager = ChatTaskManager()
        task = ChatTask(session_id="s2", asyncio_task=MagicMock())
        manager._tasks["s2"] = task
        assert manager.get_task("s2") is task

    @pytest.mark.asyncio
    async def test_stop_task_missing_returns_false(self):
        """stop_task returns False for non-existent session."""
        manager = ChatTaskManager()
        assert await manager.stop_task("ghost") is False

    @pytest.mark.asyncio
    async def test_shutdown_completes_without_tasks(self):
        """shutdown completes cleanly with no tasks."""
        manager = ChatTaskManager()
        await manager.shutdown()
        assert manager._tasks == {}
        assert manager._completed_tasks == {}
        assert manager.has_active_tasks() is False

    @pytest.mark.asyncio
    async def test_wait_all_tasks_completes_without_tasks(self):
        """wait_all_tasks completes cleanly with no tasks."""
        manager = ChatTaskManager()
        await manager.wait_all_tasks()
        assert manager._tasks == {}
        assert manager._completed_tasks == {}
        assert manager.has_active_tasks() is False

    @pytest.mark.asyncio
    async def test_push_event_appends_to_buffer(self):
        """_push_event adds event to task's event list and notifies."""
        manager = ChatTaskManager()
        task = ChatTask(session_id="s3", asyncio_task=MagicMock())
        manager._tasks["s3"] = task

        from datus.api.models.cli_models import SSEEvent, SSEPingData

        event = SSEEvent(id=1, event="ping", data=SSEPingData(), timestamp="2025-01-01T00:00:00Z")
        await manager._push_event(task, event)
        assert len(task.events) == 1
        assert task.events[0] is event

    @pytest.mark.asyncio
    async def test_stop_task_with_no_node_cancels_asyncio_task(self):
        """stop_task cancels asyncio task when node is not set."""
        manager = ChatTaskManager()
        mock_asyncio_task = MagicMock()
        mock_asyncio_task.done.return_value = False
        task = ChatTask(session_id="s4", asyncio_task=mock_asyncio_task)
        task.node = None
        manager._tasks["s4"] = task

        result = await manager.stop_task("s4")
        assert result is True
        mock_asyncio_task.cancel.assert_called_once()


@pytest.mark.asyncio
class TestStartChat:
    """Tests for start_chat — background task creation."""

    async def test_start_chat_creates_task(self, real_agent_config, mock_llm_create):
        """start_chat creates a ChatTask and returns it."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello", session_id="start-test")
        task = await manager.start_chat(real_agent_config, request)
        assert task is not None
        assert task.session_id == "start-test"
        assert task.status == "running"
        # Clean up
        await manager.shutdown()

    async def test_start_chat_duplicate_session_raises(self, real_agent_config, mock_llm_create):
        """start_chat raises ValueError for duplicate session_id."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello", session_id="dup-session")
        await manager.start_chat(real_agent_config, request)

        with pytest.raises(ValueError, match="already running"):
            await manager.start_chat(real_agent_config, StreamChatInput(message="again", session_id="dup-session"))
        await manager.shutdown()

    async def test_start_chat_generates_session_id(self, real_agent_config, mock_llm_create):
        """start_chat generates session_id when not provided."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello")
        task = await manager.start_chat(real_agent_config, request)
        assert task.session_id is not None
        assert len(task.session_id) > 0
        await manager.shutdown()

    async def test_start_chat_with_subagent(self, real_agent_config, mock_llm_create):
        """start_chat with sub_agent_id creates task with correct session pattern."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello")
        task = await manager.start_chat(real_agent_config, request, sub_agent_id="gen_sql")
        assert "gen_sql" in task.session_id
        await manager.shutdown()

    async def test_start_chat_fills_database_context(self, real_agent_config, mock_llm_create):
        """start_chat fills database context from request."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello", database="california_schools")
        task = await manager.start_chat(real_agent_config, request)
        assert task is not None
        assert real_agent_config.current_database == "california_schools"
        await manager.shutdown()

    async def test_stop_running_task_with_node(self, real_agent_config, mock_llm_create):
        """stop_task interrupts a running task that has a node set."""
        from unittest.mock import MagicMock

        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello", session_id="stop-test")
        task = await manager.start_chat(real_agent_config, request)

        # Set a mock node with interrupt_controller
        mock_node = MagicMock()
        mock_node.interrupt_controller.interrupt = MagicMock()
        task.node = mock_node

        result = await manager.stop_task("stop-test")
        assert result is True
        mock_node.interrupt_controller.interrupt.assert_called_once()
        await manager.shutdown()

    async def test_wait_all_tasks_with_running(self, real_agent_config, mock_llm_create):
        """wait_all_tasks waits for running tasks without cancelling."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="wait test", session_id="wait-test")
        task = await manager.start_chat(real_agent_config, request)

        # wait_all_tasks should return (tasks may finish quickly with mock LLM)
        await manager.wait_all_tasks()
        assert task.asyncio_task.done() is True
        assert manager._tasks == {}
        assert manager.get_task("wait-test") is task
        assert manager.has_active_tasks() is False
        await manager.shutdown()

    async def test_consume_events_yields_ping_when_idle(self, monkeypatch):
        """consume_events yields a ping event when idle past HEARTBEAT_INTERVAL."""
        from datus.api.models.cli_models import SSEEvent, SSEPingData
        from datus.api.services import chat_task_manager as ctm

        monkeypatch.setattr(ctm, "HEARTBEAT_INTERVAL", 0.05)

        manager = ChatTaskManager()
        task = ChatTask(session_id="ping-test", asyncio_task=MagicMock())
        task.status = "running"
        manager._tasks["ping-test"] = task

        gen = manager.consume_events(task, start_from=0)

        # First yield should be a ping triggered by timeout
        first = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert first.event == "ping"

        # Now push a real event and ensure it is consumed next
        real_event = SSEEvent(id=1, event="message", data=SSEPingData(), timestamp="2025-01-01T00:00:00Z")
        await manager._push_event(task, real_event)

        nxt = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        # May get another ping first if timing races; loop until we see the real event
        while nxt.event == "ping":
            nxt = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert nxt.event == "message"
        assert nxt.id == 1

        # Mark task done so generator exits
        async with task.condition:
            task.status = "completed"
            task.condition.notify_all()
        await gen.aclose()

    async def test_consume_events_from_completed_task(self, real_agent_config, mock_llm_create):
        """consume_events yields buffered events from completed task."""
        from datus.api.models.cli_models import SSEEvent, SSEPingData

        manager = ChatTaskManager()
        task = ChatTask(session_id="consume-test", asyncio_task=MagicMock())
        task.status = "completed"
        event = SSEEvent(id=1, event="ping", data=SSEPingData(), timestamp="2025-01-01T00:00:00Z")
        task.events = [event]
        manager._tasks["consume-test"] = task

        events = []
        async for e in manager.consume_events(task, start_from=0):
            events.append(e)
        assert len(events) == 1
        assert events[0].id == 1


class TestResolveAtContext:
    """Tests for _resolve_at_context — @ reference resolution."""

    def test_resolve_empty_paths_returns_empty(self, real_agent_config):
        """_resolve_at_context with no paths returns empty lists."""
        manager = ChatTaskManager()
        tables, metrics, sqls = manager._resolve_at_context(real_agent_config, None, None, None)
        assert tables == []
        assert metrics == []
        assert sqls == []

    def test_resolve_with_empty_lists(self, real_agent_config):
        """_resolve_at_context with empty lists returns empty results."""
        manager = ChatTaskManager()
        tables, metrics, sqls = manager._resolve_at_context(real_agent_config, [], [], [])
        assert tables == []
        assert metrics == []
        assert sqls == []

    def test_resolve_nonexistent_paths(self, real_agent_config):
        """_resolve_at_context with nonexistent paths returns empty (no crash)."""
        manager = ChatTaskManager()
        tables, metrics, sqls = manager._resolve_at_context(
            real_agent_config,
            ["nonexistent/table/path"],
            ["nonexistent/metric/path"],
            ["nonexistent/sql/path"],
        )
        # Should return empty lists since paths don't exist
        assert isinstance(tables, list)
        assert isinstance(metrics, list)
        assert isinstance(sqls, list)


class TestCreateNode:
    """Tests for _create_node — agentic node factory."""

    def test_create_gen_sql_node(self, real_agent_config, mock_llm_create):
        """_create_node creates GenSQLAgenticNode for gen_sql."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_sql", "test-session")
        assert isinstance(node, GenSQLAgenticNode)

    def test_create_node_returns_agentic_node(self, real_agent_config, mock_llm_create):
        """_create_node returns an AgenticNode subclass for any valid subagent_id."""
        from datus.agent.node.agentic_node import AgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "chat", "test-session")
        assert isinstance(node, AgenticNode)

    def test_create_default_node_for_none(self, real_agent_config, mock_llm_create):
        """_create_node creates an AgenticNode when subagent_id is None."""
        from datus.agent.node.agentic_node import AgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, None, "test-session")
        assert isinstance(node, AgenticNode)

    def test_create_gen_semantic_model_node(self, real_agent_config, mock_llm_create):
        """_create_node creates GenSemanticModelAgenticNode for gen_semantic_model."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_semantic_model", "test-session")
        assert isinstance(node, GenSemanticModelAgenticNode)

    def test_create_gen_metrics_node(self, real_agent_config, mock_llm_create):
        """_create_node creates GenMetricsAgenticNode for gen_metrics."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_metrics", "test-session")
        assert isinstance(node, GenMetricsAgenticNode)

    def test_create_gen_ext_knowledge_node(self, real_agent_config, mock_llm_create):
        """_create_node creates GenExtKnowledgeAgenticNode for gen_ext_knowledge."""
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_ext_knowledge", "test-session")
        assert isinstance(node, GenExtKnowledgeAgenticNode)


class TestCreateNodeInput:
    """Tests for _create_node_input — input model factory."""

    def test_gen_sql_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for GenSQLAgenticNode returns GenSQLNodeInput."""
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_sql", "test")
        result = manager._create_node_input("test query", node, [], [], [])
        assert isinstance(result, GenSQLNodeInput)
        assert result.user_message == "test query"

    def test_default_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for default node returns valid input."""
        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, None, "test")
        result = manager._create_node_input("hello", node, [], [], [])
        assert result.user_message == "hello"

    def test_semantic_model_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for GenSemanticModelAgenticNode returns SemanticNodeInput."""
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_semantic_model", "test")
        result = manager._create_node_input("generate model", node, [], [], [])
        assert isinstance(result, SemanticNodeInput)

    def test_metrics_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for GenMetricsAgenticNode returns SemanticNodeInput."""
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_metrics", "test")
        result = manager._create_node_input("generate metrics", node, [], [], [])
        assert isinstance(result, SemanticNodeInput)

    def test_ext_knowledge_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for GenExtKnowledgeAgenticNode returns ExtKnowledgeNodeInput."""
        from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_ext_knowledge", "test")
        result = manager._create_node_input("extract knowledge", node, [], [], [])
        assert isinstance(result, ExtKnowledgeNodeInput)

    def test_sql_summary_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for SqlSummaryAgenticNode returns SqlSummaryNodeInput."""
        from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_sql_summary", "test")
        result = manager._create_node_input("summarize sql", node, [], [], [])
        assert isinstance(result, SqlSummaryNodeInput)

    def test_node_input_with_db_context(self, real_agent_config, mock_llm_create):
        """_create_node_input passes database context through."""

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_sql", "test")
        result = manager._create_node_input(
            "test",
            node,
            [],
            [],
            [],
            catalog="cat",
            database="db",
            db_schema="schema",
        )
        assert result.catalog == "cat"
        assert result.database == "db"
        assert result.db_schema == "schema"

    def test_feedback_node_input_with_source_session(self, real_agent_config, mock_llm_create):
        """_create_node_input for FeedbackAgenticNode carries source_session_id through."""
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.schemas.feedback_agentic_node_models import FeedbackNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "feedback", "test")
        assert isinstance(node, FeedbackAgenticNode)

        result = manager._create_node_input(
            '[The user reacted to this message "reply" with [thumbsup]]',
            node,
            [],
            [],
            [],
            database="db",
            source_session_id="chat_session_xyz",
        )
        assert isinstance(result, FeedbackNodeInput)
        assert result.source_session_id == "chat_session_xyz"
        assert result.database == "db"


# ---------------------------------------------------------------------------
# Helpers for thinking-delta tests
# ---------------------------------------------------------------------------


def _make_thinking_delta(event_id: int, text: str, message_id: str = "m1", data_type=SSEDataType.APPEND_MESSAGE):
    """Create a thinking-delta SSEEvent."""
    return SSEEvent(
        id=event_id,
        event="message",
        data=SSEMessageData(
            type=data_type,
            payload=SSEMessagePayload(
                message_id=message_id,
                role="assistant",
                content=[IMessageContent(type="thinking", payload={"content": text})],
            ),
        ),
        timestamp="2025-01-01T00:00:00Z",
    )


def _make_markdown_event(event_id: int, text: str, message_id: str = "m1"):
    """Create a non-delta markdown message SSEEvent."""
    return SSEEvent(
        id=event_id,
        event="message",
        data=SSEMessageData(
            type=SSEDataType.APPEND_MESSAGE,
            payload=SSEMessagePayload(
                message_id=message_id,
                role="assistant",
                content=[IMessageContent(type="markdown", payload={"content": text})],
            ),
        ),
        timestamp="2025-01-01T00:00:00Z",
    )


def _make_ping_event(event_id: int = -1):
    return SSEEvent(id=event_id, event="ping", data=SSEPingData(), timestamp="2025-01-01T00:00:00Z")


# ---------------------------------------------------------------------------
# _is_thinking_delta tests
# ---------------------------------------------------------------------------


class TestIsThinkingDelta:
    """Tests for _is_thinking_delta identification."""

    def test_positive_append(self):
        """Correctly identifies APPEND_MESSAGE thinking delta."""
        ev = _make_thinking_delta(0, "hello")
        assert _is_thinking_delta(ev) is True

    def test_positive_create(self):
        """Correctly identifies CREATE_MESSAGE thinking delta."""
        ev = _make_thinking_delta(0, "hello", data_type=SSEDataType.CREATE_MESSAGE)
        assert _is_thinking_delta(ev) is True

    def test_negative_markdown(self):
        """Markdown content is not a thinking delta."""
        ev = _make_markdown_event(0, "hello")
        assert _is_thinking_delta(ev) is False

    def test_negative_update_message(self):
        """UPDATE_MESSAGE is not a thinking delta."""
        ev = SSEEvent(
            id=0,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.UPDATE_MESSAGE,
                payload=SSEMessagePayload(
                    message_id="m1",
                    role="assistant",
                    content=[IMessageContent(type="thinking", payload={"content": "x"})],
                ),
            ),
            timestamp="t",
        )
        assert _is_thinking_delta(ev) is False

    def test_negative_ping_event(self):
        """Ping event is not a thinking delta."""
        ev = _make_ping_event()
        assert _is_thinking_delta(ev) is False

    def test_negative_empty_content(self):
        """Empty content list is not a thinking delta."""
        ev = SSEEvent(
            id=0,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.APPEND_MESSAGE,
                payload=SSEMessagePayload(message_id="m1", role="assistant", content=[]),
            ),
            timestamp="t",
        )
        assert _is_thinking_delta(ev) is False


# ---------------------------------------------------------------------------
# _coalesce_deltas tests
# ---------------------------------------------------------------------------


class TestCoalesceDeltas:
    """Tests for _coalesce_deltas batch merging."""

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _coalesce_deltas([]) == []

    def test_single_delta(self):
        """Single delta is returned unchanged."""
        ev = _make_thinking_delta(0, "hi")
        result = _coalesce_deltas([ev])
        assert len(result) == 1
        assert result[0] is ev  # same object, no copy needed

    def test_merges_consecutive(self):
        """3 consecutive deltas merge into 1 with concatenated text."""
        evts = [_make_thinking_delta(i, f"part{i}") for i in range(3)]
        result = _coalesce_deltas(evts)
        assert len(result) == 1
        merged = result[0]
        assert merged.id == 0  # retains first event's id
        data = merged.data
        assert isinstance(data, SSEMessageData)
        assert data.payload.content[0].payload["content"] == "part0part1part2"

    def test_preserves_non_delta(self):
        """Non-delta events pass through unchanged."""
        md = _make_markdown_event(0, "hello")
        ping = _make_ping_event(1)
        result = _coalesce_deltas([md, ping])
        assert len(result) == 2
        assert result[0] is md
        assert result[1] is ping

    def test_mixed_sequence(self):
        """delta + non-delta + delta → 3 events (non-delta breaks run)."""
        d1 = _make_thinking_delta(0, "a")
        md = _make_markdown_event(1, "break")
        d2 = _make_thinking_delta(2, "b")
        result = _coalesce_deltas([d1, md, d2])
        assert len(result) == 3
        # First is the lone delta (unchanged)
        assert result[0] is d1
        # Second is the markdown
        assert result[1] is md
        # Third is the lone delta (unchanged)
        assert result[2] is d2

    def test_trailing_delta_run(self):
        """Non-delta followed by multiple deltas → 2 events."""
        md = _make_markdown_event(0, "start")
        d1 = _make_thinking_delta(1, "x")
        d2 = _make_thinking_delta(2, "y")
        result = _coalesce_deltas([md, d1, d2])
        assert len(result) == 2
        assert result[0] is md
        data = result[1].data
        assert isinstance(data, SSEMessageData)
        assert data.payload.content[0].payload["content"] == "xy"

    def test_different_message_ids_break_run(self):
        """Consecutive deltas with different message_ids are NOT merged."""
        d1 = _make_thinking_delta(0, "a", message_id="m1")
        d2 = _make_thinking_delta(1, "b", message_id="m1")
        d3 = _make_thinking_delta(2, "c", message_id="m2")
        d4 = _make_thinking_delta(3, "d", message_id="m2")
        result = _coalesce_deltas([d1, d2, d3, d4])
        # Should produce 2 merged events (one per message_id)
        assert len(result) == 2
        data0 = result[0].data
        data1 = result[1].data
        assert isinstance(data0, SSEMessageData)
        assert isinstance(data1, SSEMessageData)
        assert data0.payload.message_id == "m1"
        assert data0.payload.content[0].payload["content"] == "ab"
        assert data1.payload.message_id == "m2"
        assert data1.payload.content[0].payload["content"] == "cd"


# ---------------------------------------------------------------------------
# Integration: consume_events with coalescing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConsumeEventsCoalescing:
    """Integration test: queued deltas are coalesced during consumption."""

    async def test_consume_events_coalesces_queued_deltas(self, monkeypatch):
        """Multiple queued thinking deltas are yielded as a single merged event."""
        from datus.api.services import chat_task_manager as ctm

        monkeypatch.setattr(ctm, "HEARTBEAT_INTERVAL", 0.05)

        manager = ChatTaskManager()
        task = ChatTask(session_id="coalesce-test", asyncio_task=MagicMock())
        task.status = "running"
        manager._tasks["coalesce-test"] = task

        # Push 3 thinking deltas while consumer is not running
        for i in range(3):
            await manager._push_event(task, _make_thinking_delta(i, f"chunk{i}"))

        # Mark done so consumer exits after draining
        async with task.condition:
            task.status = "completed"
            task.condition.notify_all()

        events = []
        async for e in manager.consume_events(task, start_from=0):
            events.append(e)

        # Should receive 1 merged event instead of 3
        assert len(events) == 1
        data = events[0].data
        assert isinstance(data, SSEMessageData)
        assert data.payload.content[0].payload["content"] == "chunk0chunk1chunk2"

        # cursor should have advanced past all 3 original events
        assert task.consumer_offset == 3


@pytest.mark.asyncio
class TestRunLoopPathManagerContext:
    """Regression: _run_loop must pin agent_config.path_manager into its own context.

    The gateway bridge dispatches messages from a Feishu SDK worker thread via
    ``asyncio.run_coroutine_threadsafe``. That thread never inherited the
    ContextVar set by ``AgentConfig.__init__``, so the spawned task starts
    with an empty ``_current_path_manager`` and downstream stores
    (``BaseSubjectEmbeddingStore`` -> ``get_subject_tree_store``) would fall
    back to a path manager with empty ``project_name``, raising
    ``create_rdb_for_store requires a non-empty project``.
    """

    async def test_run_loop_sets_path_manager_when_context_is_empty(self, real_agent_config, mock_llm_create):
        """Even when the calling context has no path manager, _run_loop binds
        agent_config.path_manager so downstream get_path_manager() callers see
        the right project_name."""
        from datus.api.models.cli_models import StreamChatInput
        from datus.utils.path_manager import (
            _current_path_manager,
            get_path_manager,
            reset_path_manager,
        )

        captured: dict[str, str] = {}

        original_create_node = ChatTaskManager._create_node

        def _capturing_create_node(self, agent_config, subagent_id, session_id, **kwargs):
            # Simulate what BaseSubjectEmbeddingStore.__init__ does at line 692:
            # rely on the ambient ContextVar to find the project name.
            captured["project_name"] = get_path_manager().project_name
            return original_create_node(self, agent_config, subagent_id, session_id, **kwargs)

        manager = ChatTaskManager()
        manager._create_node = _capturing_create_node.__get__(manager, ChatTaskManager)  # type: ignore[method-assign]

        # Wipe the ContextVar to mimic the Feishu-thread dispatch case. The
        # task created by start_chat will inherit this empty context.
        token = _current_path_manager.set(None)
        try:
            assert _current_path_manager.get() is None
            request = StreamChatInput(message="hello", session_id="path-mgr-test")
            task = await manager.start_chat(real_agent_config, request)
            # Wait for the background loop to finish (mock LLM returns immediately).
            await asyncio.wait_for(task.asyncio_task, timeout=5.0)
        finally:
            reset_path_manager(token)
            await manager.shutdown()

        expected = real_agent_config.project_name
        assert expected, "real_agent_config.project_name must be non-empty for this test"
        assert captured.get("project_name") == expected, (
            f"_run_loop did not pin path_manager: got {captured.get('project_name')!r}, expected {expected!r}"
        )

    async def test_run_loop_does_not_leak_path_manager_to_caller(self, real_agent_config, mock_llm_create):
        """The ContextVar.set inside _run_loop must stay scoped to its own task
        and not bleed back into the calling context."""
        from datus.api.models.cli_models import StreamChatInput
        from datus.utils.path_manager import _current_path_manager, reset_path_manager

        manager = ChatTaskManager()
        token = _current_path_manager.set(None)
        try:
            request = StreamChatInput(message="hello", session_id="leak-test")
            task = await manager.start_chat(real_agent_config, request)
            await asyncio.wait_for(task.asyncio_task, timeout=5.0)
            # Caller's view stays untouched: the spawned task has its own context.
            assert _current_path_manager.get() is None
        finally:
            reset_path_manager(token)
            await manager.shutdown()


class _StubGenSQLNode:
    def __init__(self, **kwargs):
        self.node_name = kwargs.get("node_name")
        self.kwargs = kwargs


class TestCreateNodeCustomSubAgent:
    """Tests for _create_node custom sub_agent branch — node_name resolution."""

    def test_custom_subagent_resolves_sanitized_key(self, monkeypatch):
        """Custom sub_agent UUID resolves to sanitized node_name via agentic_nodes."""
        monkeypatch.setattr(
            "datus.api.services.chat_task_manager.GenSQLAgenticNode",
            _StubGenSQLNode,
        )
        agent_config = MagicMock()
        agent_config.agentic_nodes = {
            "my_sanitized_name": {"id": "uuid-123", "system_prompt": "my_sanitized_name"},
        }
        node = ChatTaskManager()._create_node(agent_config, "uuid-123", "s1")
        assert isinstance(node, _StubGenSQLNode)
        assert node.node_name == "my_sanitized_name"

    def test_custom_subagent_unknown_falls_back_to_id(self, monkeypatch):
        """Unknown subagent_id is used as-is when no matching entry exists."""
        monkeypatch.setattr(
            "datus.api.services.chat_task_manager.GenSQLAgenticNode",
            _StubGenSQLNode,
        )
        agent_config = MagicMock()
        agent_config.agentic_nodes = {
            "my_sanitized_name": {"id": "uuid-123", "system_prompt": "my_sanitized_name"},
        }
        node = ChatTaskManager()._create_node(agent_config, "unknown", "s1")
        assert isinstance(node, _StubGenSQLNode)
        assert node.node_name == "unknown"


class TestStartChatLanguageOverride:
    """``StreamChatInput.language`` must land on the cloned config's
    ``language`` attribute so every downstream AgenticNode sees it.

    We short-circuit the async loop to avoid spinning up real nodes: the
    override happens synchronously inside ``start_chat`` before
    ``_run_loop`` is awaited.
    """

    @pytest.mark.asyncio
    async def test_request_language_overrides_cloned_config(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        real_agent_config.language = "en"
        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", language="zh")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task  # drain the fake loop
        assert captured["agent_config"].language == "zh"
        # Source config remains untouched because start_chat deep-copies.
        assert real_agent_config.language == "en"

    @pytest.mark.asyncio
    async def test_missing_language_preserves_yaml_default(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        real_agent_config.language = "zh"
        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi")  # no language field
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task
        assert captured["agent_config"].language == "zh"
