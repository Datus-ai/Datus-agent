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
        # real_agent_config has namespace "test_ns" with "california_schools" database
        _fill_database_context(real_agent_config, database="california_schools")
        assert real_agent_config.current_namespace == "test_ns"
        assert real_agent_config.current_database == "california_schools"

    def test_database_as_namespace_name(self, real_agent_config):
        """Database matching a namespace name falls back to namespace lookup."""
        _fill_database_context(real_agent_config, database="test_ns")
        assert real_agent_config.current_namespace == "test_ns"

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
