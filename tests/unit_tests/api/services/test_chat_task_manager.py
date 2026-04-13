"""Tests for datus.api.services.chat_task_manager — background task management."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from datus.api.services.chat_task_manager import (
    ChatTask,
    ChatTaskManager,
    _fill_database_context,
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
        # real_agent_config has "california_schools" in service.databases
        # After the namespace→service.databases refactor, each DB is its own namespace key
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

    @pytest.mark.asyncio
    async def test_wait_all_tasks_completes_without_tasks(self):
        """wait_all_tasks completes cleanly with no tasks."""
        manager = ChatTaskManager()
        await manager.wait_all_tasks()

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
        await manager.start_chat(real_agent_config, request)

        # wait_all_tasks should return (tasks may finish quickly with mock LLM)
        await manager.wait_all_tasks()
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
